# headless load test of a chemiscope page: chemiscope errors in the page, console warnings/errors, rendered plot nodes
C=$(ls -d ~/.cache/ms-playwright/chromium-1248/chrome-linux*/chrome | head -1)
f=$1; shift
timeout 180 $C --headless=new --no-sandbox --disable-gpu --use-gl=swiftshader --enable-unsafe-swiftshader --enable-logging=stderr --v=0 \
  --virtual-time-budget=40000 --dump-dom "file://$f$*" 2>/tmp/console.log > /tmp/dom.html
echo "$f$*"; echo "  page errors: $(grep -o 'chemiscope error: [A-Z][^<]*' /tmp/dom.html | head -2)"
echo "  console: $(grep -i -E 'CONSOLE|warn' /tmp/console.log | grep -v -i -E 'gpu|swiftshader|dbus|sandbox|fontconfig|gcm|webgl' | sed -E 's/.*CONSOLE[^"]*"//' | sort | uniq -c | head -8)"
echo "  plot nodes: $(grep -o 'main-svg' /tmp/dom.html | wc -l)  canvas: $(grep -o '<canvas' /tmp/dom.html | wc -l)"
