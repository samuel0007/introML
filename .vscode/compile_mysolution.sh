workspace=$1
reldir=${2%/*}
path="$workspace/build_gcc/$reldir"
program="$path/$(ls $path | grep '_mysolution$')"
echo "{\"program\": \"$program\"}" > ./.vscode/settings.json

make --directory $path
