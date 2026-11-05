package com.neuraldj.app;

import android.content.ContentUris;
import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;
import java.util.Collections;

/** Spleeter 4-stem separation on the phone. Writes vocals / drums / harm(bass+other) as 16-bit stereo WAV. */
@CapacitorPlugin(name = "StemSplitter")
public class StemSplitterPlugin extends Plugin {
    static final int NF = 4096, HOP = 1024, T = 512, F = 1024, BINS = 2049, PAD = NF - HOP;
    static final int SEG = T * HOP, SEGLEN = SEG + PAD;
    static final String[] ST = {"vocals", "drums", "bass", "other"};
    static final String[] GN = {"vocals", "drums", "harm"};
    static final String BASE = "https://huggingface.co/Best-Practice/spleeter-4stems-onnx/resolve/main/";
    static final double[] W = new double[NF], CS = new double[NF / 2], SN = new double[NF / 2];
    static {
        for (int i = 0; i < NF; i++) W[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NF);
        for (int i = 0; i < NF / 2; i++) { CS[i] = Math.cos(2 * Math.PI * i / NF); SN[i] = -Math.sin(2 * Math.PI * i / NF); }
    }

    static void fft(double[] re, double[] im, boolean inv) {
        int n = NF;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) { double t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
        }
        for (int len = 2; len <= n; len <<= 1) {
            int step = n / len, h = len >> 1;
            for (int i = 0; i < n; i += len) for (int k = 0; k < h; k++) {
                double wr = CS[k * step], wi = inv ? -SN[k * step] : SN[k * step];
                int a = i + k, b = a + h;
                double xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
            }
        }
        if (inv) for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
    }

    File modelDir() { return new File(getContext().getFilesDir(), "models"); }
    boolean modelsReady() {
        for (String s : ST) if (!new File(modelDir(), s + ".fp16.onnx").exists()) return false;
        return true;
    }

    @PluginMethod
    public void status(PluginCall call) {
        JSObject r = new JSObject();
        r.put("ready", modelsReady());
        call.resolve(r);
    }

    @PluginMethod
    public void download(PluginCall call) {
        new Thread(() -> {
            try {
                modelDir().mkdirs();
                for (int k = 0; k < 4; k++) {
                    File f = new File(modelDir(), ST[k] + ".fp16.onnx");
                    if (f.exists()) continue;
                    File tmp = new File(modelDir(), ST[k] + ".part");
                    HttpURLConnection h = (HttpURLConnection) new URL(BASE + ST[k] + ".fp16.onnx").openConnection();
                    h.setInstanceFollowRedirects(true);
                    h.setConnectTimeout(20000);
                    h.setReadTimeout(30000);
                    try (InputStream in = h.getInputStream(); FileOutputStream o = new FileOutputStream(tmp)) {
                        byte[] b = new byte[65536];
                        int r; long tot = 0, len = Math.max(1, h.getContentLengthLong());
                        while ((r = in.read(b)) > 0) {
                            o.write(b, 0, r); tot += r;
                            JSObject p = new JSObject();
                            p.put("pct", (int) ((k * 100L + tot * 100L / len) / 4));
                            notifyListeners("downloadProgress", p);
                        }
                    }
                    if (!tmp.renameTo(f)) throw new Exception("Could not save model");
                }
                call.resolve();
            } catch (Throwable e) { call.reject("Download failed: " + e); }
        }).start();
    }

    @PluginMethod
    public void split(PluginCall call) {
        final long id = call.getData().optLong("id", -1);
        if (id < 0) { call.reject("Missing id"); return; }
        if (!modelsReady()) { call.reject("Models not downloaded"); return; }
        new Thread(() -> {
            try { call.resolve(doSplit(id)); }
            catch (Throwable e) { call.reject("Split failed: " + e); }
        }).start();
    }

    float[][] decode(Uri uri) throws Exception {
        MediaExtractor ex = new MediaExtractor();
        ex.setDataSource(getContext(), uri, null);
        int ti = -1; MediaFormat fm = null;
        for (int i = 0; i < ex.getTrackCount(); i++) {
            MediaFormat f = ex.getTrackFormat(i);
            String m = f.getString(MediaFormat.KEY_MIME);
            if (m != null && m.startsWith("audio/")) { ti = i; fm = f; break; }
        }
        if (ti < 0) throw new Exception("No audio track");
        ex.selectTrack(ti);
        MediaCodec cd = MediaCodec.createDecoderByType(fm.getString(MediaFormat.KEY_MIME));
        cd.configure(fm, null, null, 0);
        cd.start();
        int ch = fm.getInteger(MediaFormat.KEY_CHANNEL_COUNT), sr = fm.getInteger(MediaFormat.KEY_SAMPLE_RATE);
        short[] buf = new short[1 << 20]; int cnt = 0;
        MediaCodec.BufferInfo bi = new MediaCodec.BufferInfo();
        boolean inDone = false, outDone = false;
        while (!outDone) {
            if (!inDone) {
                int ii = cd.dequeueInputBuffer(10000);
                if (ii >= 0) {
                    ByteBuffer ib = cd.getInputBuffer(ii);
                    int sz = ex.readSampleData(ib, 0);
                    if (sz < 0) { cd.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inDone = true; }
                    else { cd.queueInputBuffer(ii, 0, sz, ex.getSampleTime(), 0); ex.advance(); }
                }
            }
            int oi = cd.dequeueOutputBuffer(bi, 10000);
            if (oi >= 0) {
                ByteBuffer ob = cd.getOutputBuffer(oi);
                ob.position(bi.offset); ob.limit(bi.offset + bi.size);
                ShortBuffer sb = ob.order(ByteOrder.nativeOrder()).asShortBuffer();
                int n = sb.remaining();
                if (cnt + n > buf.length) buf = Arrays.copyOf(buf, Math.max(buf.length * 2, cnt + n));
                sb.get(buf, cnt, n); cnt += n;
                cd.releaseOutputBuffer(oi, false);
                if ((bi.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) outDone = true;
            } else if (oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                MediaFormat of = cd.getOutputFormat();
                ch = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT); sr = of.getInteger(MediaFormat.KEY_SAMPLE_RATE);
            }
        }
        cd.stop(); cd.release(); ex.release();
        int frames = cnt / ch, n2 = (int) ((long) frames * 44100 / sr);
        float[][] x = new float[2][n2];
        double ratio = (double) sr / 44100;
        for (int i = 0; i < n2; i++) {
            double p = i * ratio; int i0 = (int) p; double fr = p - i0;
            int i1 = Math.min(frames - 1, i0 + 1);
            for (int c = 0; c < 2; c++) {
                int cc = ch > 1 ? c : 0;
                double a = buf[i0 * ch + cc], b = buf[i1 * ch + cc];
                x[c][i] = (float) ((a + (b - a) * fr) / 32768.0);
            }
        }
        return x;
    }

    static void wavHeader(BufferedOutputStream o, int n) throws Exception {
        ByteBuffer h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
        h.put("RIFF".getBytes()).putInt(36 + n * 4).put("WAVEfmt ".getBytes()).putInt(16)
         .putShort((short) 1).putShort((short) 2).putInt(44100).putInt(44100 * 4).putShort((short) 4).putShort((short) 16)
         .put("data".getBytes()).putInt(n * 4);
        o.write(h.array());
    }

    JSObject doSplit(long id) throws Exception {
        Uri uri = ContentUris.withAppendedId(Build.VERSION.SDK_INT >= 29
                ? MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL)
                : MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, id);
        float[][] x = decode(uri);
        final int n = x[0].length;
        File od = new File(getContext().getFilesDir(), "stems/" + id);
        od.mkdirs();
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions so = new OrtSession.SessionOptions();
        so.setIntraOpNumThreads(4);
        OrtSession[] ses = new OrtSession[4];
        for (int k = 0; k < 4; k++) ses[k] = env.createSession(new File(modelDir(), ST[k] + ".fp16.onnx").getAbsolutePath(), so);

        BufferedOutputStream[] wo = new BufferedOutputStream[3];
        for (int g = 0; g < 3; g++) { wo[g] = new BufferedOutputStream(new FileOutputStream(new File(od, GN[g] + ".wav")), 1 << 16); wavHeader(wo[g], n); }

        final int N = (PAD + n + HOP - 1) / HOP, segs = (N + T - 1) / T;
        float[][] sre = new float[2][T * BINS], sim = new float[2][T * BINS];
        float[][] est = new float[4][2 * T * F];
        FloatBuffer inb = ByteBuffer.allocateDirect(4 * 2 * T * F).order(ByteOrder.nativeOrder()).asFloatBuffer();
        float[][][] acc = new float[3][2][SEGLEN];
        float[] wsum = new float[SEGLEN];
        double[] re = new double[NF], im = new double[NF];
        double[][] m = new double[3][F];

        for (int s = 0; s < segs; s++) {
            final int nf = Math.min(T, N - s * T);
            for (int i = 0; i < 2 * T * F; i++) inb.put(i, 0f);
            for (int c = 0; c < 2; c++) for (int f = 0; f < nf; f++) {
                int start = (s * T + f) * HOP;
                for (int i = 0; i < NF; i++) {
                    int pi = start + i - PAD;
                    re[i] = (pi >= 0 && pi < n ? x[c][pi] : 0f) * W[i]; im[i] = 0;
                }
                fft(re, im, false);
                for (int k = 0; k < BINS; k++) { sre[c][f * BINS + k] = (float) re[k]; sim[c][f * BINS + k] = (float) im[k]; }
                for (int k = 0; k < F; k++) inb.put((c * T + f) * F + k, (float) Math.hypot(re[k], im[k]));
            }
            for (int k = 0; k < 4; k++) {
                inb.rewind();
                try (OnnxTensor t = OnnxTensor.createTensor(env, inb, new long[]{2, 1, T, F});
                     OrtSession.Result r = ses[k].run(Collections.singletonMap("x", t))) {
                    FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
                    fb.rewind(); fb.get(est[k]);
                }
            }
            for (int c = 0; c < 2; c++) for (int f = 0; f < nf; f++) {
                double[] mean = new double[3];
                for (int k = 0; k < F; k++) {
                    int ix = (c * T + f) * F + k;
                    double e0 = est[0][ix] * (double) est[0][ix], e1 = est[1][ix] * (double) est[1][ix],
                           e2 = est[2][ix] * (double) est[2][ix], e3 = est[3][ix] * (double) est[3][ix];
                    double tot = e0 + e1 + e2 + e3 + 1e-10;
                    m[0][k] = (e0 + 2.5e-11) / tot; m[1][k] = (e1 + 2.5e-11) / tot;
                    m[2][k] = (e2 + e3 + 5e-11) / tot;
                    for (int g = 0; g < 3; g++) mean[g] += m[g][k];
                }
                for (int g = 0; g < 3; g++) {
                    mean[g] /= F;
                    for (int k = 0; k < BINS; k++) {
                        double mk = k < F ? m[g][k] : mean[g];
                        re[k] = sre[c][f * BINS + k] * mk; im[k] = sim[c][f * BINS + k] * mk;
                    }
                    for (int k = 1; k < NF / 2; k++) { re[NF - k] = re[k]; im[NF - k] = -im[k]; }
                    fft(re, im, true);
                    int at = f * HOP;
                    for (int i = 0; i < NF; i++) acc[g][c][at + i] += (float) (re[i] * W[i]);
                }
            }
            for (int f = 0; f < nf; f++) for (int i = 0; i < NF; i++) wsum[f * HOP + i] += (float) (W[i] * W[i]);

            boolean last = s == segs - 1;
            int len = last ? SEGLEN : SEG;
            ByteBuffer ob = ByteBuffer.allocate(4 * len).order(ByteOrder.LITTLE_ENDIAN);
            for (int g = 0; g < 3; g++) {
                ob.clear();
                int w = 0;
                for (int i = 0; i < len; i++) {
                    long gi = (long) s * SEG + i - PAD;
                    if (gi < 0 || gi >= n) continue;
                    float ws = wsum[i];
                    for (int c = 0; c < 2; c++) {
                        float v = ws > 1e-8f ? acc[g][c][i] / ws : 0f;
                        ob.putShort((short) Math.max(-32767, Math.min(32767, Math.round(v * 32767f))));
                    }
                    w += 4;
                }
                wo[g].write(ob.array(), 0, w);
            }
            if (!last) {
                for (int g = 0; g < 3; g++) for (int c = 0; c < 2; c++) {
                    System.arraycopy(acc[g][c], SEG, acc[g][c], 0, PAD);
                    Arrays.fill(acc[g][c], PAD, SEGLEN, 0f);
                }
                System.arraycopy(wsum, SEG, wsum, 0, PAD);
                Arrays.fill(wsum, PAD, SEGLEN, 0f);
            }
            JSObject p = new JSObject();
            p.put("pct", (s + 1) * 100 / segs);
            notifyListeners("splitProgress", p);
        }
        for (int g = 0; g < 3; g++) wo[g].close();
        for (OrtSession o : ses) o.close();
        JSObject ret = new JSObject();
        ret.put("dir", od.getAbsolutePath());
        return ret;
    }
}
