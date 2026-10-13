import { chromium } from '/opt/npm-tools/node_modules/playwright/index.mjs';
const url=process.argv[2], out=process.argv[3];
const b=await chromium.launch({executablePath:'/opt/pw-browsers/chromium-1194/chrome-linux/chrome',args:['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
const p=await b.newPage({viewport:{width:1300,height:800}});
p.on('console',m=>console.log('console.'+m.type()+':',m.text().slice(0,400)));
p.on('pageerror',e=>console.log('pageerror:',e.message.slice(0,400)));
await p.goto(url);
try{ await p.waitForFunction(()=>window.__ready&&window.__ready(),null,{timeout:90000}); console.log('READY'); }catch(e){ console.log('NOT READY:', await p.evaluate(()=>document.getElementById('lmsg')?.textContent+' | '+document.getElementById('err')?.textContent)); }
await p.evaluate(()=>window.__setView("plaza",true)); await p.waitForTimeout(3000); await p.screenshot({path:out}); await b.close();
