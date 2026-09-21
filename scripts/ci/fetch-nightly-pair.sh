#!/usr/bin/env bash
#
# Fetch the nightly linux wheel pair into a dist dir, ready for scripts/publish-wheels.sh.
#
# Usage:
#   scripts/ci/fetch-nightly-pair.sh [dist-dir]       (default: ./dist)
#
# Source is the rolling nightly release (its engine-linux_x86_64.json names the pair), or
# with RUN_ID the `wheels` artifact of one nightly-wheels run, which GitHub keeps for 7
# days -- the way to promote an earlier night. Release downloads are checked against the
# manifest's sha256 and size; every wheel's filename stamp is checked against the source
# commit and, when given, EXPECT_COMMIT.
#
# Environment:
#   NIGHTLY_REPO    repo holding the nightly release (default: FlashML-org/FreeToken)
#   NIGHTLY_TAG     its tag (default: nightly)
#   RUN_ID          nightly-wheels run id to take the artifact from instead
#   EXPECT_COMMIT   sha prefix (7-40 hex) the pair must have been built from
set -euo pipefail

REPO="${NIGHTLY_REPO:-FlashML-org/FreeToken}"
TAG="${NIGHTLY_TAG:-nightly}"
RUN_ID="${RUN_ID:-}"
EXPECT="${EXPECT_COMMIT:-}"
DIST="${1:-dist}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

for tool in gh jq curl sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if [ -n "$EXPECT" ]; then
  [[ "$EXPECT" =~ ^[0-9a-f]{7,40}$ ]] || die "EXPECT_COMMIT must be 7-40 hex chars (got '$EXPECT')"
fi
# Stamps are 9 chars, EXPECT_COMMIT may be shorter or the full sha.
same_commit() { [[ "$1" == "$2"* || "$2" == "$1"* ]]; }

mkdir -p "$DIST"
[ -z "$(ls -A "$DIST")" ] || die "$DIST is not empty"

if [ -n "$RUN_ID" ]; then
  [[ "$RUN_ID" =~ ^[0-9]+$ ]] || die "RUN_ID must be a plain number (got '$RUN_ID')"
  run_json="$(gh api "repos/$REPO/actions/runs/$RUN_ID")" || die "cannot read run $RUN_ID on $REPO"
  workflow="$(jq -r .name <<<"$run_json")"
  branch="$(jq -r .head_branch <<<"$run_json")"
  sha="$(jq -r .head_sha <<<"$run_json")"
  conclusion="$(jq -r .conclusion <<<"$run_json")"
  [ "$workflow" = "Nightly wheels" ] || die "run $RUN_ID is '$workflow', not a Nightly wheels run"
  [ "$branch" = main ] || die "run $RUN_ID built branch '$branch', not main"
  [ "$conclusion" = success ] || die "run $RUN_ID concluded '$conclusion', not success"
  if [ -n "$EXPECT" ]; then
    same_commit "$sha" "$EXPECT" || die "run $RUN_ID built ${sha:0:9}, not $EXPECT"
  fi
  # A night whose HEAD was already published succeeds without building, so a green
  # run is not proof of an artifact; check for it and its 7-day expiry explicitly.
  artifact="$(gh api "repos/$REPO/actions/runs/$RUN_ID/artifacts" \
    --jq '.artifacts[] | select(.name == "wheels") | if .expired then "expired" else "ok" end')"
  case "$artifact" in
    ok) ;;
    expired) die "run $RUN_ID's wheels artifact has expired (7-day retention)" ;;
    *) die "run $RUN_ID has no wheels artifact -- a night whose HEAD was already published builds nothing" ;;
  esac
  say "downloading the 'wheels' artifact of run $RUN_ID (${sha:0:9})"
  gh run download "$RUN_ID" -R "$REPO" -n wheels -D "$DIST"
  source_commit="${sha:0:9}"
else
  url="https://github.com/$REPO/releases/download/$TAG/engine-linux_x86_64.json"
  manifest="$(curl -fsSL "$url")" || die "cannot fetch $url"
  source_commit="$(jq -r '.commit // empty' <<<"$manifest")"
  [ -n "$source_commit" ] || die "the manifest at $url names no commit"
  if [ -n "$EXPECT" ]; then
    same_commit "$source_commit" "$EXPECT" || die "$TAG currently holds $source_commit, not $EXPECT"
  fi
  say "$TAG holds $source_commit (published $(jq -r .published <<<"$manifest"))"
  # The manifest is data from the network; only a plain wheel filename served from
  # this very release is acceptable, whatever the JSON says.
  prefix="https://github.com/$REPO/releases/download/$TAG/"
  for part in runtime kernel_cache; do
    name="$(jq -r ".$part.name" <<<"$manifest")"
    part_url="$(jq -r ".$part.url" <<<"$manifest")"
    want_sha="$(jq -r ".$part.sha256" <<<"$manifest")"
    want_size="$(jq -r ".$part.size" <<<"$manifest")"
    [[ "$name" =~ ^freetoken[_-][A-Za-z0-9._+-]+\.whl$ ]] || die "manifest names a suspicious asset: '$name'"
    [[ "$part_url" == "$prefix"* ]] || die "manifest URL is not under $prefix: $part_url"
    [[ "$want_sha" =~ ^[0-9a-f]{64}$ && "$want_size" =~ ^[0-9]+$ ]] || die "manifest has a malformed sha256/size for $part"
    say "downloading $name"
    curl -fsSL -o "$DIST/$name" "$part_url" || die "download failed: $part_url"
    got_size="$(wc -c <"$DIST/$name" | tr -d ' ')"
    [ "$got_size" = "$want_size" ] || die "$name: size $got_size, manifest says $want_size"
    got_sha="$(sha256sum "$DIST/$name" | cut -d' ' -f1)"
    [ "$got_sha" = "$want_sha" ] || die "$name: sha256 does not match the manifest"
  done
fi

# The filename stamp is what the publisher pairs and the consumers see; it must agree
# with the source no matter how the wheels were fetched.
shopt -s nullglob
wheels=("$DIST"/*.whl)
[ "${#wheels[@]}" -eq 2 ] || die "expected exactly 2 wheels in $DIST, found ${#wheels[@]}"
for w in "${wheels[@]}"; do
  stamp="$(grep -oE '[+.]g[0-9a-f]{7,}' <<<"${w##*/}" | head -1 | sed 's/^[+.]g//' || true)"
  [ -n "$stamp" ] || die "$(basename "$w") carries no build stamp"
  same_commit "$stamp" "$source_commit" || die "$(basename "$w") was built from $stamp, expected $source_commit"
done

say "fetched into $DIST:"
ls -l "$DIST"
