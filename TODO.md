# TODO
- [ ] Add dynamic arrays in grids (start with somewhat smaller cells-arrays and increase their size if necessary). Useful to decrease a little bit the memory usage and hopefully it does not decrease the performance too much. See the [grid](lppydsmc/data_structures/grid.py).
- [x] Add config files and readers (and think first about what things ought to be in config files and what others things should not)
- [ ] OPTIONAL : Better plotting package. Maybe use [bokeh](https://docs.bokeh.org/en/latest/docs/gallery.html). Maybe solutions already exist online. Check with LPPIC team for a better plotting module too.
- [ ] add code dependencies
- [ ] allow tracking of out_particles at each frame ? (by default, and not allow any other thing ! - there is no reason for that one, including the wall collisions etc)
- [x] background gas with lower dt (so the collisions with  walls happens closer to the walls)
- [ ] Split the main script into several other smallers scripts so it is less a pain to maintain and navigate trought (and understand for newcomers)
- [ ] Better documentation in the README