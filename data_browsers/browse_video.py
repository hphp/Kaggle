import cv2
import sys
import filters
from managers import WindowManager, CaptureManager
import rects
from trackers import FaceTracker

class Browser(object):
    
    def __init__(self,video_source):  
        self._windowManager = WindowManager('Browser', self.onKeypress)
        self._captureManager = CaptureManager(video_source, self._windowManager, True)
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False
        self._curveFilter = filters.BGRPortraCurveFilter()
    
    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            
            if frame is not None:
                print "got frame" 
                self._faceTracker.update(frame)
                faces = self._faceTracker.faces
                rects.swapRects(frame, frame,
                                [face.faceRect for face in faces])
            
                #filters.strokeEdges(frame, frame)
                #self._curveFilter.apply(frame, frame)
                
                if self._shouldDrawDebugRects:
                    self._faceTracker.drawDebugRects(frame)
            else:
                print "got None frame"
                print "press any key to exit."
                cv2.waitKey(0)
                break
            self._captureManager.exitFrame()
            waitkey_time=1
            if self._captureManager._video_source!=0:   
                waitkey_time=500
            self._windowManager.processEvents(waitkey_time)
    
    def onKeypress(self, keycode):
        """Handle a keypress.
        
        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        x      -> Start/stop drawing debug rectangles around faces.
        escape -> Quit.
        
        """
        if keycode == 32: # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo(
                    '/Users/xcbfreedom/Documents/screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 120: # x
            self._shouldDrawDebugRects = \
                not self._shouldDrawDebugRects
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()


if __name__=="__main__":
    if len(sys.argv)!=2:
        print   ''' USAGE: %s [<image_file_name> | camera]''' % sys.argv[0]
        print   '''x-- do detect;'''
        print   '''tab---generate a video;'''
        print   '''space--take a screenshot;'''
        print   '''escape--exit;'''
        sys.exit(0)

    if sys.argv[1]=='camera':
        # do the camera
        browser = Browser(0)
    else:
        # do the video file 
        #filename='/Users/xcbfreedom/Documents/video.avi'
        #filename='/Users/xcbfreedom/Documents/screencast.avi'
        filename=sys.argv[1]
        browser = Browser(filename)

    browser.run() # 
