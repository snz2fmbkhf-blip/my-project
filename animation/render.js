// 把 index.html 渲染成 MP4（1920×1080，30fps，无音轨）。
// 用法：node animation/render.js [输出路径] [--subs]
//   默认不画字幕（配音后用剪映“自动字幕”生成）；--subs 则保留网页里的字幕。
// 逐帧调用 anim.seek(t) 截图，经管道送入 ffmpeg，画面与浏览器播放完全一致。
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const path = require('path');

const FPS = 30;
const args = process.argv.slice(2);
const withSubs = args.includes('--subs');
const out = path.resolve(args.find(a => !a.startsWith('--')) || path.join(__dirname, 'export', 'bp-regulation.mp4'));

(async () => {
  require('fs').mkdirSync(path.dirname(out), { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await page.goto('file://' + path.join(__dirname, 'index.html') + '?clean');
  if (!withSubs) await page.addStyleTag({ content: '#subs { display: none; }' });
  const { T0, T1 } = await page.evaluate(() => ({ T0: anim.T0, T1: anim.T1 }));
  const frames = Math.round((T1 - T0) * FPS);

  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
    { stdio: ['pipe', 'inherit', 'inherit'] });
  const done = new Promise((res, rej) => ff.on('close', c => (c === 0 ? res() : rej(new Error('ffmpeg exit ' + c)))));

  for (let i = 0; i < frames; i++) {
    await page.evaluate(t => anim.seek(t), T0 + i / FPS);
    const buf = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log(`frame ${i}/${frames}`);
  }
  ff.stdin.end();
  await done;
  await browser.close();
  console.log('done:', out);
})().catch(e => { console.error(e); process.exit(1); });
