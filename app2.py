import asyncio
import json
import logging
import os
import ssl
import cv2
import uuid
from aiohttp import web
from av import VideoFrame
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

import os.path as path
import sys
import traceback

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from fastseg import MobileV3Large, MobileV3Small
from fastseg.image import colorize, blend




##Load Self signed SSL certificate
context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
context.load_cert_chain(certfile="server.crt",keyfile="server.key")


##Get Current Working Directory
ROOT = os.path.dirname(__file__)


##Declarations
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

torch.backends.cudnn.benchmark = True
model = MobileV3Large.from_pretrained(None, num_filters=128).cuda().eval()


##Class containing functions to process video frames
class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        pil_img = frame.to_image().resize((480,480),resample=0)

        #Semantic Segmentation of image
        seg = model.predict_one(pil_img)
        colorized = colorize(seg)
        composited = blend(pil_img, colorized)

        #convert image into cv2 image array
        arr_img = np.array(colorized)
        cv2_img = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)
        print(cv2_img)

        #Get image Properties
        height, width, channels = cv2_img.shape

        #Get image center point
        cx,cy=width//2,height//2
        ProjectionPoint = (height//2)

        #Draw motion projection lines
        cv2.line(cv2_img, (0,height), (cx,ProjectionPoint), (50, 168, 153), 3)
        cv2.line(cv2_img, (width,height), (cx,ProjectionPoint), (50, 168, 153), 3)
        cv2.line(cv2_img, (cx,ProjectionPoint), (cx,0),(50,168,153,3))
        
        
        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(cv2_img, format='rgb24')
                
        
        
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    
        


##Render function to return index.html
async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


##Render javascript file to client
async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)



##Create SDP(Session Description Protocol) offer
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    
    #Logger Function
    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    
    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track)
                )
            )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


##Connection Close Handler
async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 443))
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host='0.0.0.0', port=port, ssl_context= context
    )


