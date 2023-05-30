## To Do

- [ ] Remove images from repo, allow GitHub to handle
- [ ] Better docs for both usage and training (includes style numbering)
- [ ] `Handwrite` Wrapper for `hand` (includes textwrap.wrap, basic decisions for user)
- [ ] Capitilize class names
- [ ] Add type hints
- [ ] Web GUI
- [ ] Text align (left, center, right) | for left select any: 
  - `strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2 # Comment `
  - `strokes[:, 0] += (view_width + random.randint(-15, 15) - strokes[:, 0].min())`
  - `strokes[:, 0] += (random.randint(0, 30) - strokes[:, 0].min())`
- [ ] CLI and argparser
- [ ] CI/CD:
  - [ ] maybe python package
  - [ ] Docker image

## In Progress

- [ ] Better structure for model training (`prepare_data.py`, `data/`, `rnn.__init__`)

## Done 

- [x] Migrate to v2 
- [x] Move code under submodules  
