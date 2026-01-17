# DynamicsofCosmoinflation2021
#### Hosts the code I have written as part of my master's project - those who are interested in the context and result of this work should contact me at j.stephenson1179@gmail.com

All `.ipynb` files are readable in Jupyter, `.mw` files are readable with Maple, and filenames are as referenced in my report "Dynamics of Cosomological Inflation", included here as Cosmoinflation.pdf. Rights to distribution reserved to Jacob Stephenson.

## Updates
I've refactored the code to support extensibility to new potentials, different step adaptation strategies, and more broadly to make the codebase more modular. This has been implemented through `potentials.py` for abstractions that are used in `integrate.py` to compute numerical trajectories on any such absstracted potentials. See `Numerical HJ examples.ipynb` for example usage, and the Code Refactor Supplement pdf to read about these changes in more detail.