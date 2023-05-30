import numpy as np
import svgwrite

from handwriting_synthesis import drawing


def _draw(strokes, lines, filename, stroke_colors=None, stroke_widths=None):
    stroke_colors = stroke_colors or ['black'] * len(lines)
    stroke_widths = stroke_widths or [2] * len(lines)

    line_height = 60
    view_width = 1000
    view_height = line_height * (len(strokes) + 1)

    dwg = svgwrite.Drawing(filename=filename)
    dwg.viewbox(width=view_width, height=view_height)
    dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

    initial_coord = np.array([0, -(3 * line_height / 4)])
    for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):

        if not line:
            initial_coord[1] -= line_height
            continue

        offsets[:, :2] *= 1.5
        strokes = drawing.offsets_to_coords(offsets)
        strokes = drawing.denoise(strokes)
        strokes[:, :2] = drawing.align(strokes[:, :2])

        strokes[:, 1] *= -1
        strokes[:, :2] -= strokes[:, :2].min() + initial_coord
        strokes[:, 0] += (view_width - strokes[:, 0].max()) / 2

        prev_eos = 1.0
        p = "M{},{} ".format(0, 0)
        for x, y, eos in zip(*strokes.T):
            p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
            prev_eos = eos
        path = svgwrite.path.Path(p)
        path = path.stroke(color=color, width=width, linecap='round').fill("none")
        dwg.add(path)

        initial_coord[1] -= line_height

    dwg.save()
