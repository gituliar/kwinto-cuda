## Release Instructions

Make sure **no changes are pending** (`git status` is empty).

- Define a release version to be pasted in `src/kwVersion.h` by CMake

  ```
  git tag -m "pre-build" vX.Y.Z
  ```

  ```
  make release
  ```

  ```
  git commit -m "Release vX.Y.Z"
  ```

  ```
  git tag -f -m "Release vX.Y.Z" vX.Y.Z
  ```

  ```
  git push --follow-tags
  ```
