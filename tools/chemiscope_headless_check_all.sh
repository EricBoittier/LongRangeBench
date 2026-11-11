# headless test of every dataset of the combined viewer (#<dataset>): page errors + screenshot per dataset
C=$(ls -d ~/.cache/ms-playwright/chromium-1248/chrome-linux*/chrome | head -1); H=$1; OUT=$2; mkdir -p $OUT
for k in combined dimers trimers families_fixedq emulation tauri_sapt10k_charged; do
  timeout 180 $C --headless=new --no-sandbox --use-gl=swiftshader --enable-unsafe-swiftshader --window-size=1400,1000 \
    --virtual-time-budget=45000 --dump-dom "file://$H#$k" 2>/dev/null > /tmp/dom_$k.html
  timeout 180 $C --headless=new --no-sandbox --use-gl=swiftshader --enable-unsafe-swiftshader --window-size=1400,1000 \
    --virtual-time-budget=45000 --screenshot=$OUT/shot_$k.png "file://$H#$k" 2>/dev/null >/dev/null
  echo "$k: errors=[$(grep -o 'chemiscope error: [A-Z][^<]*' /tmp/dom_$k.html | head -1)] shot=$(stat -c %s $OUT/shot_$k.png)"
done
