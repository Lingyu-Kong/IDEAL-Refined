from __future__ import annotations

import os

os.system(
    "python generate_with_promoter.py --nanoparticle_element Ru --promoter_element Cs --add_promoter_method semi_embedded"
)
os.system(
    "python generate_with_promoter.py --nanoparticle_element Ru --promoter_element Cs --add_promoter_method on_surface"
)
os.system(
    "python generate_with_promoter.py --np_size 1600 --nanoparticle_element Ru --promoter_element Cs --add_promoter_method semi_embedded"
)
os.system(
    "python generate_with_promoter.py --np_size 1600 --nanoparticle_element Ru --promoter_element Cs --add_promoter_method on_surface"
)
os.system(
    "python generate_with_promoter.py --np_size 2000 --nanoparticle_element Ru --promoter_element Cs --add_promoter_method semi_embedded"
)
os.system(
    "python generate_with_promoter.py --np_size 2000 --nanoparticle_element Ru --promoter_element Cs --add_promoter_method on_surface"
)
