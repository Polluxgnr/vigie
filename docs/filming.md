# Filming your own test clips

The public datasets contain skeletons only, so the demo video comes from footage you record
yourself. Staged scenes with friends who agree to be filmed are enough. Nobody needs to be
recognisable: Vigie renders a pixelated privacy view by default.

## Setup

- A phone on a shelf or a tripod, **high up (2 to 2.5 m) and angled down**, like a store camera.
  Landscape, 1080p, 30 fps is fine; Vigie resamples to 15 fps.
- A "shelf": a bookcase, a kitchen counter, or a real shop with the owner's permission.
- Put 10 to 20 small items on it (cans, bottles, snack bars).
- Good, even lighting. Full bodies visible from head to knees at least.
- Wear different clothes across takes: hoodie, jacket with pockets, t-shirt, backpack.

## Shot list (about 30 clips of 20 to 40 s)

**Normal shopping (half of the clips)**, so we can measure false alarms:
1. Browse, pick an item, look at it, put it back.
2. Pick an item and put it in a basket or a tote bag held openly.
3. Take a phone out of a pocket, use it, put it back (the classic false alarm).
4. Scratch your stomach, adjust your jacket, put hands in pockets while browsing.
5. Two people chatting and browsing together.
6. Crouch to look at the bottom shelf.

**Staged concealment (other half)**:
7. Pick an item, look around, slip it into a trouser pocket.
8. Pick an item, slip it into a jacket inside pocket.
9. Pick an item, hide it under a t-shirt or hoodie at the waistband.
10. Pick an item, drop it into an open backpack or handbag.
11. Same as 7 to 10 but turned away from the camera (harder).
12. One person stands in front as a screen while the other conceals.

Write down, for each clip, the second at which the concealment happens. These timestamps are
the ground truth for the demo's event-level check.

## Running Vigie on a clip

```bash
vigie run path/to/clip.mp4                    # writes runs/demo/clip.mp4 and clip.events.jsonl
vigie export                                  # optional: OpenVINO model, faster on Intel CPUs
vigie run clip.mp4 --pose-model yolo11n-pose_openvino_model/
```

Do not publish footage of anyone who has not agreed to it, and never film real customers.
