/* Run the browser engine (assets/cvf-live.js) headless in node and print its canon as JSON.
 *   node materials/cvf_node_canon.js assets/cvf-sample.js assets/cvf-live.js > canon-node.json */
const fs = require("fs"), vm = require("vm");
const ctx = { window: {}, Buffer, Math, Date, Object, Array, JSON, Float64Array, Uint8Array, Infinity, module: { exports: {} } };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
vm.runInContext(fs.readFileSync(process.argv[3], "utf8"), ctx);
const t0 = Date.now();
const out = ctx.module.exports.canon(ctx.window.CVF_SAMPLE);
out._ms = Date.now() - t0;
process.stdout.write(JSON.stringify(out, null, 1));
