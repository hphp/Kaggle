import cv2
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
            self._windowManager.processEvents()
    
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
    # do the video file 
    filename='/Users/xcbfreedom/Documents/video.avi'
    #filename='/Users/xcbfreedom/Documents/screencast.avi'
    #browser = Browser(filename)
    
    # do the camero
    browser = Browser(0)

    browser.run() # 
