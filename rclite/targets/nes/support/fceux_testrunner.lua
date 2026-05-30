-- FCEUX headless test runner for the rclite NES target.
--
-- Speaks the same de-facto NES test protocol (blargg) that Mesen's
-- --testrunner does: it polls the result region in PRG-RAM at $6000, and once
-- the protocol signature ($6001-$6003 = DE B0 61) is present and the status
-- byte drops below $80 (test finished), it prints the NUL-terminated message
-- at $6004 to stdout and exits with the status byte as the process exit code
-- (0 == pass).
--
-- Run headlessly (FCEUX is a Qt app, so wrap in a virtual X server):
--   xvfb-run -a fceux --loadlua fceux_testrunner.lua rc.nes
--
-- A frame budget bounds how long we wait for the test to signal completion.

local MAX_FRAMES = tonumber(os.getenv("RCLITE_MAX_FRAMES")) or 1800

-- Run the emulator unthrottled (otherwise FCEUX advances at ~60 fps, so a
-- multi-second on-device compute would take that long in wall-clock too).
if emu.speedmode then emu.speedmode("maximum") end

local function read_msg()
  local s, a = "", 0x6004
  while a <= 0x7fff do
    local c = memory.readbyte(a)
    if c == 0 then break end
    s = s .. string.char(c)
    a = a + 1
  end
  return s
end

local function finish(code, text)
  io.write(text)
  io.flush()
  os.exit(code)
end

local frame = 0
while frame < MAX_FRAMES do
  local sig = memory.readbyte(0x6001) == 0xDE
          and memory.readbyte(0x6002) == 0xB0
          and memory.readbyte(0x6003) == 0x61
  if sig then
    local st = memory.readbyte(0x6000)
    if st < 0x80 then               -- 0x00..0x7F: test finished, st == exit code
      finish(st, read_msg())
    end
  end
  emu.frameadvance()
  frame = frame + 1
end

finish(99, "TIMEOUT\n")
