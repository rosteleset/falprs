const registered_faces = {1: "Albert Einstein"}
const http_port = 12347;
const tz_offset = (new Date()).getTimezoneOffset() * 60000; //offset in milliseconds

const axios = require("axios");
const http = require("http");
const fs = require('fs');
require('log-timestamp')(function() {
    return '[' + new Date(Date.now() - tz_offset).toISOString().slice(0, -1).replace('T', ' ') + ']';
});
const { createCanvas, loadImage } = require('canvas')

// Test
const screenshot_path = "/tmp/frs_backend/screenshots/";
const frs_api_url = "http://localhost:9051/frs/api/";

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
            var txt = '';
            if ('faceId' in obj) {
                let id_face = obj.faceId;
                if (registered_faces[id_face] !== undefined) {
                    txt = registered_faces[id_face]
                    console.log(`This is ${txt}`);
                    // for a example, open the door here
                }
            }

            var s_path = screenshot_path;
            const r = {
                method: 'post',
                url: frs_api_url + 'bestQuality',
                headers: {'Accept': 'application/json', 'Content-Type': 'application/json'},
                data: {eventId: obj.eventId}
            };
            axios(r).then(async function (response) {
                // get image from FRS
                loadImage(response.data.data.screenshotUrl).then((image) => {
                    const canvas = createCanvas(image.width, image.height);
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(image, 0, 0);
                    var parts = response.data.data.screenshotUrl.split("/");
                    const filename = parts[parts.length - 1];
                    const prefix = filename.split(".")[0];

                    ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
                    ctx.lineWidth = 3;
                    ctx.beginPath();

                    ctx.moveTo(response.data.data.left, response.data.data.top + response.data.data.height / 5);
                    ctx.lineTo(response.data.data.left, response.data.data.top);
                    ctx.lineTo(response.data.data.left + response.data.data.width / 5, response.data.data.top);

                    ctx.moveTo(response.data.data.left + response.data.data.width, response.data.data.top + response.data.data.height / 5);
                    ctx.lineTo(response.data.data.left + response.data.data.width, response.data.data.top);
                    ctx.lineTo(response.data.data.left + response.data.data.width - response.data.data.width / 5, response.data.data.top);

                    ctx.moveTo(response.data.data.left, response.data.data.top + response.data.data.height - response.data.data.height / 5);
                    ctx.lineTo(response.data.data.left, response.data.data.top + response.data.data.height);
                    ctx.lineTo(response.data.data.left + response.data.data.width / 5, response.data.data.top + response.data.data.height);

                    ctx.moveTo(response.data.data.left + response.data.data.width, response.data.data.top + response.data.data.height - response.data.data.height / 5);
                    ctx.lineTo(response.data.data.left + response.data.data.width, response.data.data.top + response.data.data.height);
                    ctx.lineTo(response.data.data.left + response.data.data.width - response.data.data.width / 5, response.data.data.top + response.data.data.height);

                    ctx.stroke();

                    if (txt !== '') {
                        const fh = Math.floor(response.data.data.height * 0.1);
                        const margin = response.data.data.height * 0.05;
                        ctx.font = fh.toString() + "px Monospace";
                        var text = ctx.measureText(txt);
                        if (text.width > 0) {
                            ctx.fillStyle = 'rgba(27, 27, 27, 0.8)'; 
                            ctx.beginPath();
                            ctx.rect(response.data.data.left, response.data.data.top - fh - 2 * margin, text.width + 2 * margin, fh + 2 * margin - 3);
                            ctx.fill();

                            ctx.fillStyle = 'rgb(255, 102, 2)';
                            ctx.fillText(txt, response.data.data.left + margin, response.data.data.top - margin);
                        }
                    }

                    buffer = canvas.toBuffer("image/jpeg");
                    console.log(`Save image to file: ${s_path}${prefix}.jpg`);
                    fs.writeFileSync(`${s_path}${prefix}.jpg`, buffer);
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
