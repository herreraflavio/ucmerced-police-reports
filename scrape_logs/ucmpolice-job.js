const fs = require("fs");
fs.appendFileSync("/home/hacker/scripts/ucmpolice-job.log", `Ran at ${new Date().toISOString()}\n`);
