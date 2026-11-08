package io.github.c0rwin.coreset.experiments.systems;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import org.apache.spark.executor.TaskMetrics;
import org.apache.spark.scheduler.SparkListener;
import org.apache.spark.scheduler.SparkListenerExecutorAdded;
import org.apache.spark.scheduler.SparkListenerExecutorRemoved;
import org.apache.spark.scheduler.SparkListenerStageSubmitted;
import org.apache.spark.scheduler.SparkListenerTaskEnd;
import org.apache.spark.scheduler.SparkListenerTaskStart;

/**
 * Task metrics summed per job group (one group per repetition): shuffle bytes written and read (local plus remote),
 * task result bytes sent to the driver, executor CPU and run time, failed task attempts. Executor membership changes
 * are kept with their timestamps to time recovery from a killed executor, and a kill can be armed to fire a given time
 * after the first task of a repetition started, naming an executor that is busy then (so a running task is lost).
 */
final class SparkStats extends SparkListener {

    static final class Agg {
        long tasks;
        long failedTasks;
        long shuffleWrite;
        long shuffleRead;
        long resultSize;
        long executorCpuNanos;
        long executorRunMillis;
        long gcMillis;
        final Set<Integer> stages = ConcurrentHashMap.newKeySet();
    }

    private record ExecutorEvent(long time, String id, boolean added, String reason) {
    }

    private final Map<Integer, String> stageGroup = new ConcurrentHashMap<>();
    private final Map<String, Agg> groups = new ConcurrentHashMap<>();
    private final CopyOnWriteArrayList<ExecutorEvent> executors = new CopyOnWriteArrayList<>();
    private final Map<Long, String> running = new ConcurrentHashMap<>(); // task id -> executor id, armed group only
    private final AtomicBoolean started = new AtomicBoolean();
    private final AtomicBoolean fired = new AtomicBoolean();
    private final AtomicLong armedCompleted = new AtomicLong();
    private final ScheduledExecutorService timer = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "kill-executor");
        t.setDaemon(true);
        return t;
    });
    private volatile String armedGroup;
    private volatile long armedDelayMs;
    private volatile BiConsumer<String, Long> onTrigger;

    /**
     * Calls {@code action(executorId, finishedTasks)} once, {@code delayMs} after the first task of {@code group}
     * started, naming an executor running a task of the group at that moment (nothing happens if none is).
     */
    void armKill(String group, long delayMs, BiConsumer<String, Long> action) {
        running.clear();
        started.set(false);
        fired.set(false);
        armedCompleted.set(0);
        armedDelayMs = delayMs;
        onTrigger = action;
        armedGroup = group;
    }

    void disarm() {
        armedGroup = null;
    }

    /** True once the armed action ran (it records its own outcome). */
    boolean fired() {
        return fired.get();
    }

    private boolean armedFor(int stageId) {
        String g = armedGroup;
        return g != null && g.equals(stageGroup.get(stageId));
    }

    private void fire() {
        java.util.Iterator<String> busy = running.values().iterator();
        if (armedGroup != null && busy.hasNext() && fired.compareAndSet(false, true)) {
            onTrigger.accept(busy.next(), armedCompleted.get());
        }
    }

    @Override
    public void onStageSubmitted(SparkListenerStageSubmitted e) {
        String g = e.properties() == null ? null : e.properties().getProperty("spark.jobGroup.id");
        if (g != null) {
            stageGroup.put(e.stageInfo().stageId(), g);
        }
    }

    @Override
    public void onTaskStart(SparkListenerTaskStart e) {
        if (armedFor(e.stageId())) {
            running.put(e.taskInfo().taskId(), e.taskInfo().executorId());
            if (started.compareAndSet(false, true)) {
                timer.schedule(this::fire, armedDelayMs, TimeUnit.MILLISECONDS);
            }
        }
    }

    @Override
    public void onTaskEnd(SparkListenerTaskEnd e) {
        if (armedFor(e.stageId())) {
            running.remove(e.taskInfo().taskId());
            if (e.taskInfo().successful()) {
                armedCompleted.incrementAndGet();
            }
        }
        String g = stageGroup.get(e.stageId());
        if (g == null) {
            return;
        }
        Agg a = groups.computeIfAbsent(g, x -> new Agg());
        synchronized (a) {
            a.tasks++;
            a.stages.add(e.stageId());
            if (e.taskInfo().failed() || e.taskInfo().killed()) {
                a.failedTasks++;
            }
            TaskMetrics m = e.taskMetrics();
            if (m != null) {
                a.shuffleWrite += m.shuffleWriteMetrics().bytesWritten();
                a.shuffleRead += m.shuffleReadMetrics().remoteBytesRead() + m.shuffleReadMetrics().localBytesRead();
                a.resultSize += m.resultSize();
                a.executorCpuNanos += m.executorCpuTime();
                a.executorRunMillis += m.executorRunTime();
                a.gcMillis += m.jvmGCTime();
            }
        }
    }

    @Override
    public void onExecutorAdded(SparkListenerExecutorAdded e) {
        executors.add(new ExecutorEvent(e.time(), e.executorId(), true, ""));
    }

    @Override
    public void onExecutorRemoved(SparkListenerExecutorRemoved e) {
        executors.add(new ExecutorEvent(e.time(), e.executorId(), false, e.reason()));
    }

    /** Removes and returns the totals of a job group (empty totals if it ran no task). */
    Agg take(String group) {
        Agg a = groups.remove(group);
        return a != null ? a : new Agg();
    }

    /** Seconds from {@code killEpochMs} to the first executor removal and the first executor added after it. */
    Map<String, Object> executorEventsAfter(long killEpochMs) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (ExecutorEvent e : executors) {
            if (e.time() < killEpochMs) {
                continue;
            }
            if (!e.added() && !out.containsKey("executor_removed_s")) {
                out.put("executor_removed_s", (e.time() - killEpochMs) / 1e3);
                out.put("executor_removed_reason", e.reason());
            }
            if (e.added() && !out.containsKey("replacement_registered_s")) {
                out.put("replacement_registered_s", (e.time() - killEpochMs) / 1e3);
            }
        }
        return out;
    }
}
