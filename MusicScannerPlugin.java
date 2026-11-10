package com.neuraldj.app;

import android.Manifest;
import android.content.ContentUris;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;

import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.PermissionState;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import com.getcapacitor.annotation.Permission;
import com.getcapacitor.annotation.PermissionCallback;

import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.FileOutputStream;

@CapacitorPlugin(
    name = "MusicScanner",
    permissions = {
        @Permission(strings = { Manifest.permission.READ_MEDIA_AUDIO }, alias = "audio"),
        @Permission(strings = { Manifest.permission.READ_EXTERNAL_STORAGE }, alias = "audioLegacy")
    }
)
public class MusicScannerPlugin extends Plugin {

    private String alias() {
        return Build.VERSION.SDK_INT >= 33 ? "audio" : "audioLegacy";
    }

    private Uri collection() {
        if (Build.VERSION.SDK_INT >= 29) {
            return MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL);
        }
        return MediaStore.Audio.Media.EXTERNAL_CONTENT_URI;
    }

    @PluginMethod
    public void scan(PluginCall call) {
        if (getPermissionState(alias()) != PermissionState.GRANTED) {
            requestPermissionForAlias(alias(), call, "permissionResult");
        } else {
            doScan(call);
        }
    }

    @PermissionCallback
    private void permissionResult(PluginCall call) {
        if (getPermissionState(alias()) == PermissionState.GRANTED) {
            doScan(call);
        } else {
            call.reject("Permission to read music was denied");
        }
    }

    private void doScan(PluginCall call) {
        JSArray tracks = new JSArray();
        String[] projection = {
            MediaStore.Audio.Media._ID,
            MediaStore.Audio.Media.DISPLAY_NAME,
            MediaStore.Audio.Media.TITLE,
            MediaStore.Audio.Media.ARTIST,
            MediaStore.Audio.Media.DURATION,
            MediaStore.Audio.Media.SIZE,
            MediaStore.Audio.Media.MIME_TYPE
        };
        // Skip very short sounds (ringtones, notifications, voice notes)
        String selection = MediaStore.Audio.Media.DURATION + ">=?";
        String[] args = { "45000" };
        try (Cursor c = getContext().getContentResolver().query(
                collection(), projection, selection, args,
                MediaStore.Audio.Media.TITLE + " COLLATE NOCASE ASC")) {
            if (c != null) {
                while (c.moveToNext()) {
                    JSObject o = new JSObject();
                    o.put("id", c.getLong(0));
                    o.put("name", c.getString(1));
                    o.put("title", c.getString(2));
                    o.put("artist", c.getString(3));
                    o.put("duration", c.getLong(4));
                    o.put("size", c.getLong(5));
                    o.put("mime", c.getString(6));
                    tracks.put(o);
                }
            }
        } catch (Exception e) {
            call.reject("Could not read music: " + e.getMessage());
            return;
        }
        JSObject ret = new JSObject();
        ret.put("tracks", tracks);
        call.resolve(ret);
    }

    // Copies one song into the app's cache so the web code can read it.
    @PluginMethod
    public void copyToCache(PluginCall call) {
        long id = call.getData().optLong("id", -1);
        if (id < 0) {
            call.reject("Missing id");
            return;
        }
        String name = call.getString("name", "track.mp3");
        String ext = "";
        int dot = name.lastIndexOf('.');
        if (dot >= 0 && name.length() - dot <= 6) ext = name.substring(dot);
        try {
            File dir = new File(getContext().getCacheDir(), "neural-import");
            if (!dir.exists()) dir.mkdirs();
            File out = new File(dir, "t" + id + ext);
            Uri uri = ContentUris.withAppendedId(collection(), id);
            try (InputStream in = getContext().getContentResolver().openInputStream(uri);
                 OutputStream os = new FileOutputStream(out)) {
                if (in == null) {
                    call.reject("Could not open song");
                    return;
                }
                byte[] buf = new byte[65536];
                int n;
                while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
            }
            JSObject ret = new JSObject();
            ret.put("path", out.getAbsolutePath());
            call.resolve(ret);
        } catch (Exception e) {
            call.reject("Copy failed: " + e.getMessage());
        }
    }

    @PluginMethod
    public void cleanup(PluginCall call) {
        try {
            File dir = new File(getContext().getCacheDir(), "neural-import");
            File[] files = dir.listFiles();
            if (files != null) for (File f : files) f.delete();
        } catch (Exception ignored) {}
        call.resolve();
    }

    private OutputStream saveOut;
    private Uri saveUri;

    @PluginMethod
    public void saveBegin(PluginCall call) {
        try {
            if (Build.VERSION.SDK_INT < 29) { call.reject("Saving needs Android 10 or newer"); return; }
            android.content.ContentValues v = new android.content.ContentValues();
            v.put(MediaStore.MediaColumns.DISPLAY_NAME, call.getString("name", "mix.m4a"));
            v.put(MediaStore.MediaColumns.MIME_TYPE, call.getString("mime", "audio/mp4"));
            v.put(MediaStore.MediaColumns.RELATIVE_PATH, "Download/Neural DJ");
            v.put(MediaStore.MediaColumns.IS_PENDING, 1);
            saveUri = getContext().getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
            saveOut = getContext().getContentResolver().openOutputStream(saveUri);
            call.resolve();
        } catch (Exception e) { call.reject("Save failed: " + e.getMessage()); }
    }

    @PluginMethod
    public void saveChunk(PluginCall call) {
        try {
            saveOut.write(android.util.Base64.decode(call.getString("data", ""), android.util.Base64.DEFAULT));
            call.resolve();
        } catch (Exception e) { call.reject("Save failed: " + e.getMessage()); }
    }

    @PluginMethod
    public void saveEnd(PluginCall call) {
        try {
            saveOut.close();
            android.content.ContentValues v = new android.content.ContentValues();
            v.put(MediaStore.MediaColumns.IS_PENDING, 0);
            getContext().getContentResolver().update(saveUri, v, null, null);
            call.resolve();
        } catch (Exception e) { call.reject("Save failed: " + e.getMessage()); }
    }
}
