const registered_numbers = ["O588OA68"];
const http_port = 12346;
const tz_offset = (new Date()).getTimezoneOffset() * 60000; //offset in milliseconds

const axios = require("axios");
const http = require("http");
const fs = require('fs');
require('log-timestamp')(function() {
    return '[' + new Date(Date.now() - tz_offset).toISOString().slice(0, -1).replace('T', ' ') + ']';
});
const { createCanvas, loadImage } = require('canvas')

// Test
const screenshot_path = "/tmp/lprs_backend/screenshots/";
const lprs_api_url = "http://localhost:9051/lprs/api/";

fs.mkdirSync(screenshot_path, { recursive: true });

//http server
const http_server = http.createServer((req, res) => {
    const url_path = req.url;
    if (url_path === "/callback") {
        let body = [];
        req.on('data', (chunk) => {
            body.push(chunk);
        }).on('end', () => {
            res.writeHead(204);
            res.end();

            body = Buffer.concat(body).toString();
            console.log("Callback: " + body);

            const obj = JSON.parse(body);
            var all_plates = [];
            var allowed = [];
            if ('plates' in obj && obj.plates != null) {
                all_plates = obj.plates.map(item => `${item.type}\t${item.number}`);
                allowed = obj.plates.flatMap(item => registered_numbers.includes(item.number) ? [item.number] : []);
            }

            if (allowed.length > 0) {
                console.log("Matching registered numbers: " + allowed.join(", "));
                // for a example, open the gates here
            }

            var s_path = screenshot_path;
            const r = {
                method: 'post',
                url: lprs_api_url + 'getEventData',
                headers: {'Accept': 'application/json', 'Content-Type': 'application/json'},
                data: {eventId: obj.eventId}
            };
            axios(r).then(async function (response) {
                // get image from LPRS
                loadImage(response.data.data.screenshotUrl).then((image) => {
                    const canvas = createCanvas(image.width, image.height);
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(image, 0, 0);
                    var parts = response.data.data.screenshotUrl.split("/");
                    const filename = parts[parts.length - 1];
                    const prefix = filename.split(".")[0];

                    // draw on the image
                    response.data.data.vehicles.forEach(vehicle => {
                        ctx.strokeStyle = vehicle.isSpecial ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 0, 255, 0.8)';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.rect(vehicle.box[0], vehicle.box[1], vehicle.box[2] - vehicle.box[0] + 1, vehicle.box[3] - vehicle.box[1] + 1);
                        ctx.stroke();

                        if (vehicle.plates)
                            vehicle.plates.forEach(plate => {
                                const kpts = plate.kpts;
                                ctx.strokeStyle = 'rgb(255, 102, 2)';
                                ctx.beginPath();
                                ctx.moveTo(kpts[0], kpts[1]);
                                ctx.lineTo(kpts[2], kpts[3]);
                                ctx.lineTo(kpts[4], kpts[5]);
                                ctx.lineTo(kpts[6], kpts[7]);
                                ctx.lineTo(kpts[0], kpts[1]);
                                ctx.stroke();

                                const fh = Math.floor((plate.box[3] - plate.box[1]) * 0.65);
                                const margin = fh * 0.1;
                                ctx.font = fh.toString() + "px Monospace";
                                var text = ctx.measureText(plate.number);
                                if (text.width > 0) {
                                    ctx.fillStyle = 'rgba(127, 127, 127, 0.8)';
                                    ctx.beginPath();
                                    ctx.rect(plate.box[0], plate.box[1] - fh - 2 * margin, text.width + 2 * margin, fh + 2 * margin);
                                    ctx.fill();

                                    ctx.fillStyle = 'rgba(0, 255, 0, 1.0)';
                                    ctx.fillText(plate.number, plate.box[0] + margin, plate.box[1] - margin);
                                }
                            });
                    });

                    buffer = canvas.toBuffer("image/jpeg");
                    console.log("Save image to file: " + s_path + prefix + ".jpg");
                    fs.writeFileSync(s_path + prefix + ".jpg", buffer);
                    fs.writeFileSync(s_path + prefix + ".txt", all_plates.join("\n"));
                    console.log("\n");
                });
            }).catch(function (error) {
                console.error(error.data);
            });
        });
    }
});
http_server.listen(http_port, "localhost", () => {
    console.log(`Listening for requests on port ${http_port}\n`);
});
