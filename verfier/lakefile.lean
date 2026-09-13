import Lake
open Lake DSL

package «ToxSearchVerfier» where
  moreLeanArgs := #["-Dpp.unicode.fun=true", "-DautoImplicit=false"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.22.0"

@[default_target]
lean_lib «ToxSearch» where
  globs := #[.submodules `ToxSearch]
