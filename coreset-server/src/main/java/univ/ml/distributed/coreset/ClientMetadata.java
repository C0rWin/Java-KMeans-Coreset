package univ.ml.distributed.coreset;

import org.apache.thrift.transport.TTransport;

public class ClientMetadata {

    private String hostName;

    private int portNum;

    private TTransport socket;

    public ClientMetadata() {
    }

    public ClientMetadata(String hostName, int portNum) {
        this.hostName = hostName;
        this.portNum = portNum;
    }

    public String getHostName() {
        return hostName;
    }

    public void setHostName(String hostName) {
        this.hostName = hostName;
    }

    public int getPortNum() {
        return portNum;
    }

    public void setPortNum(int portNum) {
        this.portNum = portNum;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ClientMetadata that = (ClientMetadata) o;

        if (portNum != that.portNum) return false;
        return hostName != null ? hostName.equals(that.hostName) : that.hostName == null;

    }

    @Override
    public int hashCode() {
        int result = hostName != null ? hostName.hashCode() : 0;
        result = 31 * result + portNum;
        return result;
    }

    public void setSocket(TTransport socket) {
        this.socket = socket;
    }

    public TTransport getSocket() {
        return socket;
    }

    public void closeSocket() {
        socket.close();
    }
}
