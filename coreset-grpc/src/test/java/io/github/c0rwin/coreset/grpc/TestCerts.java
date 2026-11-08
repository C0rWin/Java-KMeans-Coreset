package io.github.c0rwin.coreset.grpc;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.GeneralSecurityException;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.Signature;
import java.security.cert.CertificateFactory;
import java.security.spec.ECGenParameterSpec;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Base64;

/**
 * Self-signed EC certificates for tests, built with the JDK alone (no keytool, no BouncyCastle): a minimal DER
 * encoder for an X.509 v3 certificate with SAN {@code DNS:localhost, IP:127.0.0.1}, usable both as a leaf and as its
 * own trust anchor.
 */
final class TestCerts {

    /** PEM files of one identity; {@code cert} doubles as the CA file that trusts it. */
    record Pem(Path cert, Path key) {
    }

    private static final byte[] OID_ECDSA_SHA256 = {0x2A, (byte) 0x86, 0x48, (byte) 0xCE, 0x3D, 0x04, 0x03, 0x02};
    private static final byte[] OID_CN = {0x55, 0x04, 0x03};
    private static final byte[] OID_SAN = {0x55, 0x1D, 0x11};
    private static final byte[] OID_BASIC_CONSTRAINTS = {0x55, 0x1D, 0x13};

    private TestCerts() {
    }

    static Pem generate(Path dir, String name) throws IOException, GeneralSecurityException {
        KeyPairGenerator kpg = KeyPairGenerator.getInstance("EC");
        kpg.initialize(new ECGenParameterSpec("secp256r1"));
        KeyPair kp = kpg.generateKeyPair();
        byte[] sigAlg = seq(tlv(0x06, OID_ECDSA_SHA256));
        byte[] subject = seq(tlv(0x31, seq(tlv(0x06, OID_CN), tlv(0x0C, name.getBytes(StandardCharsets.UTF_8)))));
        ZonedDateTime now = ZonedDateTime.now(ZoneOffset.UTC);
        DateTimeFormatter utc = DateTimeFormatter.ofPattern("yyMMddHHmmss'Z'");
        byte[] validity = seq(tlv(0x17, now.minusDays(1).format(utc).getBytes(StandardCharsets.US_ASCII)),
                tlv(0x17, now.plusDays(30).format(utc).getBytes(StandardCharsets.US_ASCII)));
        byte[] san = seq(tlv(0x82, "localhost".getBytes(StandardCharsets.US_ASCII)), tlv(0x87, new byte[] {127, 0, 0, 1}));
        byte[] extensions = tlv(0xA3, seq(
                seq(tlv(0x06, OID_SAN), tlv(0x04, san)),
                seq(tlv(0x06, OID_BASIC_CONSTRAINTS), tlv(0x01, new byte[] {(byte) 0xFF}),
                        tlv(0x04, seq(tlv(0x01, new byte[] {(byte) 0xFF}))))));
        byte[] tbs = seq(tlv(0xA0, tlv(0x02, new byte[] {2})),
                tlv(0x02, BigInteger.valueOf(System.nanoTime() & Long.MAX_VALUE).toByteArray()),
                sigAlg, subject, validity, subject, kp.getPublic().getEncoded(), extensions);
        Signature signer = Signature.getInstance("SHA256withECDSA");
        signer.initSign(kp.getPrivate());
        signer.update(tbs);
        byte[] sig = signer.sign();
        byte[] bits = new byte[sig.length + 1]; // leading 0: no unused bits
        System.arraycopy(sig, 0, bits, 1, sig.length);
        byte[] der = seq(tbs, sigAlg, tlv(0x03, bits));
        CertificateFactory.getInstance("X.509").generateCertificate(new ByteArrayInputStream(der)); // sanity check
        Path cert = dir.resolve(name + ".crt");
        Path key = dir.resolve(name + ".key");
        Files.writeString(cert, pem("CERTIFICATE", der));
        Files.writeString(key, pem("PRIVATE KEY", kp.getPrivate().getEncoded())); // PKCS#8
        return new Pem(cert, key);
    }

    private static String pem(String type, byte[] der) {
        return "-----BEGIN " + type + "-----\n" + Base64.getMimeEncoder(64, new byte[] {'\n'}).encodeToString(der)
                + "\n-----END " + type + "-----\n";
    }

    private static byte[] seq(byte[]... parts) {
        return tlv(0x30, parts);
    }

    private static byte[] tlv(int tag, byte[]... parts) {
        ByteArrayOutputStream body = new ByteArrayOutputStream();
        for (byte[] p : parts) {
            body.writeBytes(p);
        }
        int n = body.size();
        ByteArrayOutputStream out = new ByteArrayOutputStream(n + 6);
        out.write(tag);
        if (n < 0x80) {
            out.write(n);
        } else if (n < 0x100) {
            out.write(0x81);
            out.write(n);
        } else {
            out.write(0x82);
            out.write(n >>> 8);
            out.write(n);
        }
        out.writeBytes(body.toByteArray());
        return out.toByteArray();
    }
}
