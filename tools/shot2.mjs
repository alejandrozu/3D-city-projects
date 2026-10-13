import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
const views=JSON.parse(process.argv[2]); const out=process.argv[3]; const vp=JSON.parse(process.argv[4]||'{"width":1400,"height":900}');
const b=await chromium.launch({executablePath:'/opt/pw-browsers/chromium-1194/chrome-linux/chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const p=await b.newPage({viewport:vp, colorScheme: process.argv[5]||'light'});
p.on('console',m=>{ if(m.type()==='error'||m.type()==='warning') console.log('console:',m.text().slice(0,300))});
p.on('pageerror',e=>console.log('pageerror:',e.message));
await p.goto('http://localhost:8765/index.html');
await p.waitForFunction(()=>window.__ready&&window.__ready(),null,{timeout:300000});
let i=0;
for(const v of views){
  if(v.view) await p.evaluate(k=>window.__setView(k,true),v.view);
  if(v.js) await p.evaluate(v.js);
  await p.waitForTimeout(v.wait||2500);
  await p.screenshot({path:`${out}_${i++}.png`});
}
await b.close();
