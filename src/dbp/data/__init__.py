"""Dataset construction utilities: hashing, JPEG forensics and manifests.

Nothing in this package touches a model. It exists so that every dataset the project
builds carries, in a diffable CSV, exactly what it is made of: where each image came
from, what its compression history is, and a digest that proves it has not changed.
"""
