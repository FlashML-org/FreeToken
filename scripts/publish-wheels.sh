#!/usr/bin/env bash
#
# Publish a built engine wheel pair (runtime + kernel-cache) to a GitHub release.
#
# Usage:
#   scripts/publish-wheels.sh [dist-dir]       (default: ./dist)
#
# Two kinds of target share this script:
#   - the rolling `nightly` prerelease on this repo (FREETOKEN_RELEASE_ROLLING=1): created
#     on first use; after every publish its notes are rewritten from the manifest and its
#     tag is moved to FREETOKEN_RELEASE_COMMIT. Linux-only by design.
#   - a curated channel such as FreeToken-Web's `beta` release: must already exist; only
#     the wheels and the manifest are replaced, notes and tag are left alone.
#
# Deletes the release's previous wheels for each platform being published, THEN
# uploads the new ones -- in that order. Shipped Desktops resolve assets with a
# first-match scan over the release's asset list, so an old and a new wheel
# coexisting would keep serving the old one; a brief no-asset window (a clean
# install error and a retry) is the safer failure. Requires `gh` authenticated
# with write access to the target repo.
#
# After the upload, writes `engine-<platform>.json` to the release: the pair's URLs,
# sha256 and sizes under a fixed asset name, so a consumer resolves the pair with one
# static download (no api.github.com, no per-IP rate limit) and scans the asset list
# only when the manifest is missing. One file per platform -- publishing one platform
# never touches another platform's manifest.
#
# Environment:
#   FREETOKEN_RELEASE_REPO     target repo, required (e.g. FlashML-org/FreeToken)
#   FREETOKEN_RELEASE_TAG      release tag, required (e.g. nightly)
#   FREETOKEN_RELEASE_ROLLING  1 = rolling-nightly mode described above (default: 0)
#   FREETOKEN_RELEASE_COMMIT   full sha the rolling tag moves to; must match the wheel stamp
#   FREETOKEN_WEB_REPO / FREETOKEN_WEB_TAG   older names for REPO / TAG, still honored
set -euo pipefail

REPO="${FREETOKEN_RELEASE_REPO:-${FREETOKEN_WEB_REPO:-}}"
TAG="${FREETOKEN_RELEASE_TAG:-${FREETOKEN_WEB_TAG:-}}"
ROLLING="${FREETOKEN_RELEASE_ROLLING:-0}"
COMMIT="${FREETOKEN_RELEASE_COMMIT:-}"
DIST="${1:-dist}"
ROLLING_TITLE="Nightly (rolling)"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || die "gh not found"
command -v jq >/dev/null 2>&1 || die "jq not found"
[ -d "$DIST" ] || die "no such dist dir: $DIST"
{ [ -n "$REPO" ] && [ -n "$TAG" ]; } \
  || die "set FREETOKEN_RELEASE_REPO and FREETOKEN_RELEASE_TAG -- there is no default target"

case " 1 true yes on " in
  *" $(printf '%s' "$ROLLING" | tr '[:upper:]' '[:lower:]') "*) ROLLING=1 ;;
  *) ROLLING=0 ;;
esac
if [ "$ROLLING" = 1 ]; then
  [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] \
    || die "FREETOKEN_RELEASE_ROLLING needs FREETOKEN_RELEASE_COMMIT set to the full 40-hex sha the tag moves to"
  # Rolling mode force-moves its tag; a release tag must never be reachable that way.
  [[ "$TAG" != v[0-9]* ]] || die "refusing to run in rolling mode against release tag '$TAG'"
fi

shopt -s nullglob
wheels=("$DIST"/freetoken-*.whl "$DIST"/freetoken_kernel_cache-*.whl)
[ "${#wheels[@]}" -gt 0 ] || die "no freetoken wheels in $DIST"

# Platforms covered by this publish; pruning is per-platform, so a linux-only
# publish leaves the win_amd64 assets alone.
platforms="$(for w in "${wheels[@]}"; do
  case "${w##*/}" in
    *linux_x86_64*) echo linux_x86_64 ;;
    *win_amd64*) echo win_amd64 ;;
    *) echo UNKNOWN ;;
  esac
done | sort -u)"
grep -qx UNKNOWN <<<"$platforms" && die "cannot infer the platform of every wheel in $DIST"
if [ "$ROLLING" = 1 ] && [ "$platforms" != linux_x86_64 ]; then
  die "the rolling release takes the linux_x86_64 pair only (got: $(tr '\n' ' ' <<<"$platforms"))"
fi

# Publish gate: prune-then-upload replaces a platform's wheel pair WHOLESALE, so demand
# a complete, self-consistent pair up front -- exactly one runtime + one kernel-cache
# wheel per platform, with matching +g<sha> stamps when stamped. A half dist (a build
# that died between the two wheels) or a mixed-build pair must fail HERE, before any
# asset is deleted, not leave the release missing an asset it will never get back.
while IFS= read -r p; do
  rt_n=0; kc_n=0; rt_stamp=""; kc_stamp=""; kc_b=""
  for w in "${wheels[@]}"; do
    b="${w##*/}"
    case "$b" in
      freetoken-*"$p"*.whl)
        rt_n=$((rt_n + 1))
        rt_stamp="$(grep -oE '\+g[0-9a-f]{7,}' <<<"$b" | head -1 || true)"
        ;;
      freetoken_kernel_cache-*"$p"*.whl)
        kc_n=$((kc_n + 1)); kc_b="$b"
        kc_stamp="$(grep -oE '\.g[0-9a-f]{7,}' <<<"$b" | head -1 || true)"
        ;;
    esac
  done
  { [ "$rt_n" -eq 1 ] && [ "$kc_n" -eq 1 ]; } \
    || die "$DIST must hold exactly one runtime + one kernel-cache wheel for $p (found $rt_n + $kc_n)"
  if [ -n "$rt_stamp" ] && [ -n "$kc_stamp" ] && [ "${rt_stamp#+}" != "${kc_stamp#.}" ]; then
    die "stamp mismatch for $p: runtime has ${rt_stamp}, kernel-cache has ${kc_stamp} -- these wheels are from different builds"
  fi
  if [ "$ROLLING" = 1 ]; then
    [ -n "$rt_stamp" ] || die "the rolling release needs stamped wheels (+g<sha>); $p runtime wheel has no stamp"
    [[ "$COMMIT" == "${rt_stamp#+g}"* ]] \
      || die "FREETOKEN_RELEASE_COMMIT ${COMMIT:0:9} is not the commit the $p wheels were built from (${rt_stamp#+g})"
    grep -qE '\+cu[0-9]+' <<<"$kc_b" \
      || die "the rolling release needs a kernel-cache wheel with a +cuNNN version segment ($kc_b)"
  fi
done <<<"$platforms"

say "publishing to $REPO tag '$TAG':"
for w in "${wheels[@]}"; do say "  $(basename "$w")"; done

# Only the rolling release is created on demand; a curated channel that is missing
# is an operator error, not something to paper over with an empty release.
err="$(mktemp)"
if ! release_json="$(gh api "repos/$REPO/releases/tags/$TAG" 2>"$err")"; then
  grep -q 'HTTP 404' "$err" || { cat "$err" >&2; die "cannot read release '$TAG' on $REPO"; }
  [ "$ROLLING" = 1 ] || die "release '$TAG' does not exist on $REPO"
  say "creating the rolling release '$TAG' at ${COMMIT:0:9}"
  release_json="$(gh api -X POST "repos/$REPO/releases" \
    -f tag_name="$TAG" -f target_commitish="$COMMIT" -f name="$ROLLING_TITLE" \
    -f body="Automated build of main. Assets are published by scripts/publish-wheels.sh." \
    -F prerelease=true -F draft=false -f make_latest=false)"
fi
rm -f "$err"
release_id="$(jq -r '.id' <<<"$release_json")"
existing="$(jq -r '.assets[].name' <<<"$release_json")"

# Prune the previous generation first (delete-then-upload; see header).
while IFS= read -r p; do
  while IFS= read -r name; do
    case "$name" in
      freetoken-*"$p"*.whl | freetoken_kernel_cache-*"$p"*.whl)
        say "deleting old asset $name"
        gh release delete-asset "$TAG" "$name" -R "$REPO" --yes
        ;;
    esac
  done <<<"$existing"
done <<<"$platforms"

for w in "${wheels[@]}"; do
  say "uploading $(basename "$w")"
  gh release upload "$TAG" "$w" -R "$REPO"
done

# The manifest is written LAST so it never names a wheel that is not there yet. The
# Desktop compares asset basenames, so the URL is spelled the way GitHub's
# browser_download_url spells it: `+` percent-encoded.
asset_url() { printf 'https://github.com/%s/releases/download/%s/%s' "$REPO" "$TAG" "${1//+/%2B}"; }
wheel_json() {
  local w="$1" name size sha
  name="${w##*/}"
  size="$(wc -c <"$w" | tr -d ' ')"
  sha="$(sha256sum "$w" | cut -d' ' -f1)"
  printf '{"name": "%s", "url": "%s", "sha256": "%s", "size": %s}' "$name" "$(asset_url "$name")" "$sha" "$size"
}
manifest_dir="$(mktemp -d)"
trap 'rm -rf "$manifest_dir"' EXIT
while IFS= read -r p; do
  rt=""; kc=""
  for w in "${wheels[@]}"; do
    case "${w##*/}" in
      freetoken-*"$p"*.whl) rt="$w" ;;
      freetoken_kernel_cache-*"$p"*.whl) kc="$w" ;;
    esac
  done
  rt_name="${rt##*/}"
  # freetoken-<version>-<python>-<abi>-<platform>.whl
  version="$(cut -d- -f2 <<<"$rt_name")"
  python_tag="$(cut -d- -f3 <<<"$rt_name")"
  commit="$(grep -oE '\+g[0-9a-f]{7,}' <<<"$rt_name" | head -1 | sed 's/^+g//' || true)"
  cuda="$(grep -oE '\+cu[0-9]+' <<<"${kc##*/}" | head -1 | sed 's/^+//' || true)"
  manifest="$manifest_dir/engine-$p.json"
  cat >"$manifest" <<JSON
{
  "schema": 1,
  "channel": "$TAG",
  "platform": "$p",
  "version": "$version",
  "commit": "$commit",
  "published": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "python": "$python_tag",
  "cuda": "$cuda",
  "runtime": $(wheel_json "$rt"),
  "kernel_cache": $(wheel_json "$kc")
}
JSON
  say "uploading engine-$p.json (version $version)"
  gh release upload "$TAG" "$manifest" -R "$REPO" --clobber
done <<<"$platforms"

mb() { awk -v b="$1" 'BEGIN { printf "%.1f MB", b / 1000000 }'; }
render_notes() {
  local m="$1" py cuda
  py="$(jq -r .python "$m" | sed -E 's/^cp([0-9])([0-9]+)$/\1.\2/')"
  cuda="$(jq -r .cuda "$m" | sed -E 's/^cu([0-9]+)([0-9])$/\1.\2/')"
  local commit published rt_name rt_url rt_sha rt_size kc_name kc_url kc_sha kc_size
  commit="$(jq -r .commit "$m")"
  published="$(jq -r .published "$m")"
  rt_name="$(jq -r .runtime.name "$m")"; rt_url="$(jq -r .runtime.url "$m")"
  rt_sha="$(jq -r .runtime.sha256 "$m")"; rt_size="$(jq -r .runtime.size "$m")"
  kc_name="$(jq -r .kernel_cache.name "$m")"; kc_url="$(jq -r .kernel_cache.url "$m")"
  kc_sha="$(jq -r .kernel_cache.sha256 "$m")"; kc_size="$(jq -r .kernel_cache.size "$m")"
  cat <<MD
## FreeToken nightly (rolling)

Automated build of \`main\`. This release is **overwritten every night**: the \`$TAG\` tag moves to the built commit and the previous wheels are deleted. Do not pin to this tag; pin to a wheel URL below, whose filename embeds the commit it was built from.

**Install** (Linux x86_64, CPython $py, CUDA $cuda):

\`\`\`bash
uv pip install \\
  "freetoken[accel] @ $rt_url" \\
  "$kc_url"
\`\`\`

| commit | built (UTC) | python | cuda | runtime | kernel-cache |
|---|---|---|---|---|---|
| [\`$commit\`](https://github.com/$REPO/commit/$commit) | ${published:0:10} ${published:11:5} | cp${py/./} | cu${cuda/./} | $(mb "$rt_size") | $(mb "$kc_size") |

<details><summary>sha256</summary>

\`\`\`
$rt_sha  $rt_name
$kc_sha  $kc_name
\`\`\`
</details>

Machine-readable: \`engine-linux_x86_64.json\` next to the wheels (schema 1).
Stable releases: [PyPI \`freetoken\`](https://pypi.org/project/freetoken/) and the \`v*\` releases here.
MD
}

# A transient failure past this point would leave the new wheels under the old notes
# and tag, and nothing re-runs the publish; retry before giving up.
gh_retry() {
  local attempt
  for attempt in 1 2 3; do
    "$@" && return 0
    [ "$attempt" = 3 ] || sleep $((attempt * 5))
  done
  return 1
}

if [ "$ROLLING" = 1 ]; then
  notes="$manifest_dir/notes.md"
  render_notes "$manifest_dir/engine-linux_x86_64.json" >"$notes"
  say "rewriting release notes"
  gh_retry gh api -X PATCH "repos/$REPO/releases/$release_id" \
    -f name="$ROLLING_TITLE" -F prerelease=true -f make_latest=false -F body=@"$notes" >/dev/null \
    || die "cannot rewrite the notes of release '$TAG'"
  # Last, so the tag only ever points at a fully published build.
  say "moving tag '$TAG' to ${COMMIT:0:9}"
  tag_err="$manifest_dir/tag.err"
  if ! gh_retry gh api -X PATCH "repos/$REPO/git/refs/tags/$TAG" -f sha="$COMMIT" -F force=true >/dev/null 2>"$tag_err"; then
    # Only a tag deleted by hand while the release survived is recreated.
    grep -qiE 'does not exist|Not Found' "$tag_err" || { cat "$tag_err" >&2; die "cannot move tag '$TAG'"; }
    gh api -X POST "repos/$REPO/git/refs" -f ref="refs/tags/$TAG" -f sha="$COMMIT" >/dev/null \
      || die "cannot create tag '$TAG'"
  fi
fi

say "release now carries:"
gh api "repos/$REPO/releases/tags/$TAG" \
  --jq '.assets[] | select(.name | endswith(".whl")) | "  \(.name)  \(.digest // "no-digest")  \(.updated_at)"'
