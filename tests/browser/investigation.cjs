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
    const root = path.resolve(process.env.MRF_REPORT_DIR || 'examples/investigations/digits');
    const report = JSON.parse(fs.readFileSync(path.join(root, 'report.json')));
    const plan = JSON.parse(fs.readFileSync(path.join(root, 'plan.json')));
    const candidate = report.candidate || report.regressed;
    const all = candidate.slices.find(row => row.name === 'all');
    const before = new Map(JSON.parse(fs.readFileSync(path.join(root, 'execution-000.json'))).release.predictions.map(row => [row.case_id, row.observed]));
    const after = JSON.parse(fs.readFileSync(path.join(root, 'execution-001.json'))).release.predictions;
    const changedCount = after.filter(row => before.get(row.case_id) !== row.observed).length;
    await page.goto(pathToFileURL(path.join(root, process.env.MRF_REPORT_HTML || 'index.html')).href);
    const visible = page.locator('#cases tbody tr:visible');
    assert.equal(await page.inputValue('#case-state'), 'different');
    assert.equal(await visible.count(), changedCount);
    assert.equal(await page.locator('#case-count').textContent(), `${changedCount} of ${plan.cases.length} cases`);
    const overview = await page.locator('#decision-overview').textContent();
    assert(overview.includes(`Candidate policy result: ${candidate.passed ? 'Pass' : 'Fail'}`));
    assert(overview.includes(`${changedCount} changed predictions`));
    // The retained deployment is a policy pass with changed labels, not a failure.
    if (plan.declared_changes.baseline === 'float32' && plan.declared_changes.candidate === 'int8') {
      assert.equal(plan.cases.length, 1969);
      assert.equal(changedCount, 8);
      assert(overview.includes('3 new errors'));
      assert(overview.includes('3 corrections'));
      assert(overview.includes('2 changes between incorrect labels'));
      assert(overview.includes('78 of 78 declared slices passed'));
      assert(overview.includes('baseline 1734/1969'));
      assert(overview.includes('candidate 1734/1969'));
      assert(overview.includes('Historical cause: not identified'));
      assert(overview.includes('Model execution costs: unknown'));
    }
    await page.selectOption('#case-state', 'all');
    assert.equal(await visible.count(), plan.cases.length);
    const images = page.locator('#cases img');
    if (await images.count()) {
      assert.equal(await images.count(), plan.cases.length);
      assert(await images.evaluateAll(items => items.every(image =>
        image.complete && image.naturalWidth === 8 && image.naturalHeight === 8)));
    } else if (fs.existsSync(path.join(root, 'case_inputs.json'))) {
      const inputs = JSON.parse(fs.readFileSync(path.join(root, 'case_inputs.json')));
      const example = Object.values(inputs)[0].text;
      assert((await page.locator('#cases').textContent()).includes(example));
    }
    if (report.candidate) {
      assert((await page.locator('body').textContent()).includes('Repairs have not been evaluated'));
      assert.equal(await page.locator('#cases thead th').count(), 5);
    }
    await page.selectOption('#case-state', 'different');
    assert.equal(await visible.count(), changedCount);
    await page.selectOption('#case-state', 'improved');
    assert.equal(await visible.count(), all.improved_case_ids.length);
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
    assert.equal(await page.locator('#case-count').textContent(), `0 of ${plan.cases.length} cases`);
    await page.fill('#case-search', '');
    await page.selectOption('#case-state', 'all');
    assert.equal(await visible.count(), plan.cases.length);
    await page.locator('#case-inspection').evaluate(element => element.scrollIntoView());
    await page.screenshot({path: 'inspection-desktop.png'});
    await page.setViewportSize({width: 390, height: 844});
    await page.locator('#case-inspection').evaluate(element => element.scrollIntoView());
    await page.screenshot({path: 'inspection-mobile.png'});
    assert.deepEqual(errors, []);
    console.log(`${plan.cases.length} cases: previews, filters, search, empty state and JSON export passed`);
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exit(1);});
