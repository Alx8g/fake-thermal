You are judging thermal camera imagery for a photorealism study.

Each image file below shows two thermal images of the same scene side by side, labelled **A**
(left) and **B** (right). Both are coloured with the same "ironbow" thermal palette (black and
purple are cold, orange warm, yellow and white hottest) and each has its own contrast stretch.

For each file, decide which panel looks more like a genuine frame from a real thermal (LWIR,
microbolometer) camera. Think about how real scenes appear to such a camera: what is warm and
what is cold, how surfaces, sky, lights, glass, vehicles and people read, and what texture
and noise a real sensor produces. Study each pair carefully, the way an expert in thermal
imaging would. Answer every file; if you are unsure, still pick one and give a low confidence.

Rules:
- Use only the Read tool, and only on the files listed below. Do not open, search for or list
  any other files or folders, and do not run commands.
- Judge each file on its own merits. The more realistic panel is on the left about half the time.

Reply with only a JSON object, no other text, in exactly this shape:

{"001": {"realistic": "A", "confidence": 3, "why": "a few words"}, ...}

Confidence is 1 (pure guess) to 5 (certain). Use the three-digit file number as the key.

Files:
