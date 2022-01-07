import pyglet

grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

window = pyglet.window.Window(400, 500)

score_label = pyglet.text.Label(text="Score: 0", x=10, y=460)
level_label = pyglet.text.Label(text="Test", x=window.width//2, y=window.height//2, anchor_x='center')

# background = pyglet.image.SolidColorImagePattern((255, 0, 255, 255)).create_image(800, 600)


@window.event
def on_draw():
    # draw things here
    window.clear()
    # background.blit(0, 0)
    level_label.draw()
    score_label.draw()
    for row in range(4):
        for col in range(4):
            if grid[row][col] > 0:
                square = pyglet.shapes.BorderedRectangle(10 + 100 * col, 10 + 100 * row, 80, 80, border=5, color=(0, 0, 0))
                square.draw()
                number = pyglet.text.Label(text=str(2**grid[row][col]), x=50 + 100 * col, y=50 + 100 * row, anchor_x='center', anchor_y='center')
                number.draw()


def launch(mat):
    global grid
    grid = mat
    pyglet.app.run()
