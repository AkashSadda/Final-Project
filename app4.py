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

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

from fastseg import MobileV3Large, MobileV3Small
from fastseg.image import colorize, blend

#load yolo model
model = YOLO("yolov8n-seg.pt")

##load fastseg model
torch.backends.cudnn.benchmark = True
model2 = MobileV3Large.from_pretrained(None, num_filters=128).cuda().eval()


##Load Self signed SSL certificate
context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
context.load_cert_chain(certfile="server.crt",keyfile="server.key")


##Get Current Working Directory
ROOT = os.path.dirname(__file__)


##Declarations
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()
road = np.array([128, 64, 128], dtype = "uint8")
sidewalk = np.array([244, 35, 232], dtype ="uint8")
left_walk = False
center_walk = False
right_walk = False
FrameCount = 0
Hold_res = np.zeros((640, 640, 3), dtype = "uint8")
fps = 0


##Data Holder for Navigation
Result = ''
flag = {'all':False,'LeftCenter':False,'CenterRight':False,'Center':False,'Left':False,'Right':False}
Sent = False
PrevRes = ''

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
        try:
            global Result,flag,road,sidewalk,left_walk,center_walk,right_walk,FrameCount,Hold_res
            pil_img = frame.to_image().resize((640,640),resample=0)
            if True:
                #convert image into cv2 image array
                arr_img = np.array(pil_img)
                cv2_img = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)




                #Semantic Segmentation of image
                seg = model2.predict_one(pil_img)
                colorized = colorize(seg)
                composited = blend(pil_img, colorized)

                #convert image into cv2 image array
                arr_img = np.array(colorized)
                col_img = cv2.cvtColor(arr_img, cv2.COLOR_RGB2BGR)

                #object detection
                img = model(cv2_img,device=0,conf=0.7)

                #Get image Properties
                height, width, channels = cv2_img.shape


                #Get image split points
                ListenLine = (height//4)*3
                LeftCloseX,LeftCloseY = (width//4),ListenLine
                RightOpenX,RightOpenY = LeftCloseX*3,ListenLine
                
                #Annotate and Process Image
                for r in img:
            
                    annotator = Annotator(col_img)

                    #get cropped image in listen area
                    col_left = col_img[LeftCloseY:height , 0:LeftCloseX]
                    col_center = col_img[LeftCloseY:height , LeftCloseX:RightOpenX]
                    col_right = col_img[LeftCloseY:height , RightOpenX:width]

                    #identify walkable area
                    if cv2.countNonZero(cv2.inRange(col_left, road, road)) != 0 or cv2.countNonZero(cv2.inRange(col_left, sidewalk, sidewalk)) != 0:
                        left_walk = True
                    if cv2.countNonZero(cv2.inRange(col_center, road, road)) != 0 or cv2.countNonZero(cv2.inRange(col_center, sidewalk, sidewalk)) != 0:
                        center_walk = True
                    if cv2.countNonZero(cv2.inRange(col_right, road, road)) != 0 or cv2.countNonZero(cv2.inRange(col_right, sidewalk, sidewalk)) != 0:
                        right_walk = True
                    
                    boxes = r.boxes
                    for box in boxes:

                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                        c = box.cls
                        annotator.box_label(b, model.names[int(c)])
                        
                        class_id = r.names[c[0].item()]
                        cords = b.tolist()
                        cords = [round(x) for x in cords]
                        conf = round(box.conf[0].item(), 2)

                        #Listen Events
                        left, top, right, bottom = cords[0], cords[1], cords[2], cords[3]

                        #Object on Listen Line
                        if top>=ListenLine or bottom>=ListenLine:
                            #covers all three sides
                            if left<=LeftCloseX and right>=RightOpenX:
                                flag['all'] = True

                            #covers left and center
                            if left<=LeftCloseX and right<=RightOpenX:
                                flag['LeftCenter'] = True

                            #covers center and right
                            if left>=LeftCloseX and right>=RightOpenX:
                                flag['CenterRight'] = True

                            #covers only center
                            if left>=LeftCloseX and right<=RightOpenX:
                                flag['Center'] = True
                                
                            #covers only left
                            if left<=LeftCloseX and right<=LeftCloseX:
                                flag['Left'] = True

                            #covers only right
                            if left>=RightOpenX and right>=RightOpenX:
                                flag['Right'] = True
                                                                                   
                        print("Object type:", class_id)
                        print("Coordinates:", cords)
                        print("Probability:", conf)
                        print("---")
                          
                #Generate Result based on detections

                if flag['LeftCenter'] == True and flag['CenterRight'] == False:
                    if right_walk:
                        Result = "Move Right"
                    else:
                        Result = "No way Stop"
                    
                if flag['CenterRight'] == True and flag['LeftCenter'] == False:
                    if left_walk:
                        Result = "Move Left"
                    else:
                        Result = "No way Stop"
                    
                if flag['Center'] == True:
                    if flag['LeftCenter'] == True:
                        if right_walk:
                            Result = "Move Right"
                        else:
                            Result = "No way Stop"
                    if flag['CenterRight'] == True:
                        if left_walk:
                            Result = "Move Left"
                        else:
                            Result = "No way Stop"
                    if flag['Left'] == True and flag['Right'] == True:
                        Result = "No way Stop"
                    if flag['Left'] == True and flag['Right'] == False:
                        if right_walk:
                            Result = "Move Right"
                        else:
                            Result = "No way Stop"
                    if flag['Left'] == False and flag['Right'] == True:
                        if left_walk:
                            Result = "Move Left"
                        else:
                            Result = "No way Stop"
                    else:
                        if left_walk and right_walk == False:
                            Result = "Move Left"
                        elif right_walk and left_walk == False:
                            Result = "Move Right"
                        else:
                            Result = "Move Left or Right"
                        
                if flag['all'] == True or (flag['LeftCenter'] == True and flag['CenterRight'] == True):
                    Result = "No way Stop"

                #Re-initialize values for next frame
                flag = {'all':False,'LeftCenter':False,'CenterRight':False,'Center':False,'Left':False,'Right':False}
                left_walk = False
                center_walk = False
                right_walk = False

                
                cv2_img = annotator.result()  

                #Draw listen line
                cv2.line(cv2_img,(0,ListenLine),(width,ListenLine),(0, 102, 0),3)
                #Split left area
                cv2.line(cv2_img,(LeftCloseX,LeftCloseY),(LeftCloseX,height),(0, 102, 0),3)
                #Split Right area
                cv2.line(cv2_img,(RightOpenX,RightOpenY),(RightOpenX,height),(0, 102, 0),3)



                #Convert BGR image to RGB
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                
                # rebuild a VideoFrame, preserving timing information
                new_frame = VideoFrame.from_ndarray(cv2_img, format='rgb24')
                        
                
                    
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base
                FrameCount = 0
                Hold_res = new_frame
                return new_frame
            
                
                
        
        except:
            traceback.print_exc()
       
    
        


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
            global PrevRes,Result,Sent,fps
            if isinstance(message, str) and message.startswith("ping"):
                if PrevRes != Result:
                    Sent = False
                    
                if not Sent:
                    channel.send("pong " + Result)
                    PrevRes = Result
                    Sent = True
                else:
                    channel.send("pong  ")
            if isinstance(message, str) and message.startswith("fps "):
                fps = eval(message[4:])

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


