const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require('playwright');

(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const root = path.resolve('examples/investigations/digits');
    const report = JSON.parse(fs.readFileSync(path.join(root, 'report.json')));
    const plan = JSON.parse(fs.readFileSync(path.join(root, 'plan.json')));
    const all = report.regressed.slices.find(row => row.name === 'all');
    await page.goto(pathToFileURL(path.join(root, 'index.html')).href);
    const visible = page.locator('#cases tbody tr:visible');
    assert.equal(await visible.count(), 540);
    assert.equal(await page.locator('#cases img').count(), 540);
    assert(await page.locator('#cases img').evaluateAll(images =>
      images.every(image => image.complete && image.naturalWidth === 8 && image.naturalHeight === 8)));
    await page.selectOption('#case-state', 'regressed');
    assert.equal(await visible.count(), all.regressed_case_ids.length);
    const target = all.regressed_case_ids[0];
    await page.fill('#case-search', target);
    assert(await visible.count() >= 1);
    assert((await visible.allTextContents()).every(text => text.includes(target)));
    const downloadEvent = page.waitForEvent('download');
    await page.click('#export-cases');
    const download = await downloadEvent;
    const ids = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
    assert(ids.includes(target));
    assert(ids.every(id => all.regressed_case_ids.includes(id) && id.includes(target)));
    await page.fill('#case-search', 'no-such-case');
    assert.equal(await visible.count(), 0);
    assert.match(await page.locator('#case-count').textContent(), /^0 of 540 cases$/);
    await page.fill('#case-search', '');
    await page.selectOption('#case-state', 'all');
    assert.equal(await visible.count(), plan.cases.length);
    await page.locator('#case-inspection').evaluate(element => element.scrollIntoView());
    await page.screenshot({path: 'inspection-desktop.png'});
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#case-inspection').evaluate(element => element.scrollIntoView());
    await page.screenshot({path: 'inspection-mobile.png'});
    assert.deepEqual(errors, []);
    console.log('540 previews, outcome filters, search, empty state and JSON export passed');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exit(1);});
