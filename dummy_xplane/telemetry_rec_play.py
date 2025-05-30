##############################################################################
# telemetry_rec_play.py  –  for recording & playback of telemetry data
##############################################################################
import pathlib, io, itertools, typing

# ───────────────────────────── COMMON PARTS ─────────────────────────────── #
class TelemetryError(Exception):
    """Wraps all I/O-level issues so callers can `except TelemetryError` once."""

_HEADER_TEMPLATE = (
    "# interval_ms: {interval_ms}\n"
    "# vehicle: {vehicle}\n"
    "# record_count: {rec_count}\n"
    "x,y,z,roll,pitch,yaw\n"
)

# ───────────────────────────── RECORDING ────────────────────────────────── #
class TelemetryRecorder:
    """
    rec = TelemetryRecorder()
    rec.record_begin("log.csv", 16, "F-18")
    ...
    rec.update([x, y, z, roll, pitch, yaw])
    ...
    rec.close()   # or rec.abort()
    """

    def __init__(self):
        self._f: io.TextIOBase | None = None
        self._path: pathlib.Path | None = None
        self._record_counter = 0
        self._open = False

    # ── API ─────────────────────────────────────────────────────────────── #
    def record_begin(self, file_name: str, interval_ms: int, vehicle: str):
        if self._open:
            raise TelemetryError("Recorder already active; call close() first")
        self._path = pathlib.Path(file_name)
        try:
            self._f = self._path.open("w", newline="\n", encoding="utf-8")
        except OSError as e:
            raise TelemetryError(f"Cannot open '{file_name}' for writing: {e}") from e
        self._f.write(_HEADER_TEMPLATE.format(
            interval_ms=interval_ms, vehicle=vehicle, rec_count=0
        ))
        self._open = True

    def update(self, dof6: typing.Sequence[float]):
        if not self._open:
            raise TelemetryError("Recorder not started – call record_begin()")
        if len(dof6) != 6:
            raise TelemetryError("update() expects exactly 6 DOF values")
        self._f.write(",".join(map(str, dof6)) + "\n")
        self._record_counter += 1

    def close(self):
        """Patch record_count in header, then fully close file."""
        if not self._open:
            return
        self._f.flush(); self._f.close()

        # rewrite 3rd line with correct record_count
        with self._path.open("r+", encoding="utf-8") as f:
            lines = f.readlines()
            lines[2] = f"# record_count: {self._record_counter}\n"
            f.seek(0); f.writelines(lines); f.truncate()

        self._reset()

    def abort(self):
        """Throw away file in progress (if any) and release handle."""
        if self._open and self._f:
            try: self._f.close()
            finally: self._path.unlink(missing_ok=True)   # delete partial file
        self._reset()

    # ── helpers ──────────────────────────────────────────────────────────── #
    def _reset(self):
        self._f = None
        self._path = None
        self._record_counter = 0
        self._open = False


# ───────────────────────────── PLAYBACK ─────────────────────────────────── #
class TelemetryPlayer:
    """
    player = TelemetryPlayer()
    player.start_playback("log.csv")
    while True:
        rec = player.playback_service()
        if rec is None: break           # finished or aborted
        use(rec)
    player.stop_playback()              # optional explicit cleanup
    """

    def __init__(self):
        self._f: io.TextIOBase | None = None
        self._data_iter: typing.Iterator[list[float]] | None = None
        self.interval_ms: int | None = None
        self.vehicle: str | None = None
        self.record_count: int | None = None
        self._open = False

    # ── API ─────────────────────────────────────────────────────────────── #
    def start_playback(self, file_name: str):
        if self._open:
            raise TelemetryError("Playback already active; call stop_playback()")
        try:
            self._f = open(file_name, encoding="utf-8")
        except OSError as e:
            raise TelemetryError(f"Cannot open '{file_name}' for playback: {e}") from e

        # ---- parse header block ----
        headers = []
        while True:
            pos = self._f.tell()
            line = self._f.readline()
            if not line.startswith("#"):
                self._f.seek(pos)        # rewind to first data row
                break
            headers.append(line.strip())

        kv = dict(h[1:].split(":", 1) for h in headers)
        self.interval_ms = int(kv[" interval_ms"])
        self.vehicle      = kv[" vehicle"].strip()
        self.record_count = int(kv[" record_count"])

        # ---- generator over remaining rows ----
        self._data_iter = self._iter_rows()
        self._open = True

    def playback_service(self):# -> list[float] | None:
        """
        Return next record or None (EOF).  If aborted, also returns None.
        """
        if not self._open:
            raise TelemetryError("Playback not started – call start_playback()")
        try:
            return next(self._data_iter)
        except StopIteration:
            self.stop_playback()
            return None

    def stop_playback(self):
        """Close file handle and invalidate iterator; safe to call multiple times."""
        if self._open and self._f:
            self._f.close()
        self._f = None
        self._data_iter = None
        self._open = False

    # ── helpers ──────────────────────────────────────────────────────────── #
    def _iter_rows(self):
        """Internal generator reading one CSV row at a time."""
        for row in self._f:
            row = row.strip()
            if not row or row[0].isalpha():   # skip blank line or column header
                continue
            try:
                yield [float(tok) for tok in row.split(",")]
            except ValueError as e:
                raise TelemetryError(f"Malformed data row: '{row}'") from e


# ──────────────────────── basic capture harness ─────────────────────────── #
if __name__ == '__main__':
    """
    Run this file as a script to capture one CSV log of X-Plane motion data.
    
    • The CSV is named "<ICAO>_<YYYYMMDD_HHMMSS>.csv" in the current folder.
    • Recording starts automatically on the first valid telemetry frame.
    • Press any key (in the console) to stop and finalise the file.
    """
    import time, datetime, sys, os, select
    if os.name == 'nt':               # Win32: non-blocking key detection
        import msvcrt

    # --------- external dependency ------------------------------------------------
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from xplane_telemetry import XplaneTelemetry

    # --------- user-editable section ----------------------------------------------
    TELEMETRY_EVT_PORT = 10022
    SIM_IP            = "127.0.0.1"
    NORM_FACTORS      = (1, 1, 1, 1, 1, 1)          # if you apply scaling
    DEFAULT_INTERVAL_MS = 25                        # fallback if we cannot measure

    # --------- initialise objects --------------------------------------------------
    telemetry = XplaneTelemetry((SIM_IP, TELEMETRY_EVT_PORT), NORM_FACTORS)
    recorder  = TelemetryRecorder()

    recording_started = False
    first_timestamp   = None

    input("press enter key when xplane is ready")
    telemetry.send('InitComs') 
     
    print("Ready – waiting for first telemetry frame "
          "(press any key at any time to stop)…")

    try:
        while True:
            # ---- non-blocking “any key?” check ------------------------------------
            key_pressed = (
                msvcrt.kbhit() if os.name == 'nt'
                else bool(select.select([sys.stdin], [], [], 0)[0])
            )
            if key_pressed:
                if recording_started:
                    recorder.close()
                    print("Recording finished.")
                break

            # ---- grab latest sim data --------------------------------------------
            transform = telemetry.get_telemetry()       # list[6] or None
            if transform is None:
                continue                                # no data this poll

            icao = telemetry.get_icao() or "UNKNOWN"

            # ---- auto-start recording on very first frame ------------------------
            if not recording_started:
                ts                = datetime.datetime.now()
                file_name         = f"{icao}_{ts:%Y%m%d_%H%M%S}.csv"
                first_timestamp   = time.perf_counter()
                try:
                    recorder.record_begin(file_name, DEFAULT_INTERVAL_MS, icao)
                except TelemetryError as e:
                    print(f"FATAL: {e}")
                    break
                recording_started = True
                print(f"Recording ➜ {file_name}")

            # ---- write the sample ------------------------------------------------
            recorder.update(transform)

            # ---- (optional) compute actual sample interval for debug -------------
            # You can remove this block if you don't care about runtime rate.
            now = time.perf_counter()
            if first_timestamp:
                actual_ms = (now - first_timestamp) * 1000
                first_timestamp = now
                # uncomment for verbose output
                # print(f"Δt ≈ {actual_ms:6.1f} ms   {transform}")

    finally:
        # ensure everything is flushed even on Ctrl-C or unexpected error
        if recording_started:
            try: recorder.close()
            except TelemetryError as e: print(f"Warning while closing: {e}")
        telemetry.close()
        print("Shutdown complete.")

     