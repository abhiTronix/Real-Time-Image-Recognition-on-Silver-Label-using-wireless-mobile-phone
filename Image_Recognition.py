import cv2
import numpy as np
import time ,timeit
import argparse
import dlib
import datetime
import pyzbar.pyzbar as pyzbar
import platform , os , re , sys , urllib , threading, queue , shutil
import easygui
import io, types

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract



def GetMultidetectImage(detectors , modelname , image ):
    global LogoList

    start = timeit.default_timer()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    [boxes, confidences, detector_idxs] = (
        dlib.fhog_object_detector.run_multiple(detectors, image, upsample_num_times=1, adjust_threshold=0.3))
    for i in range(len(boxes)):
        x  = (boxes[i].left())
        y  = (boxes[i].top())
        xb = (boxes[i].right())
        yb = (boxes[i].bottom())
        percent = int(confidences[i] * 100)
        if percent > 100 : percent = 100

        global COLORS

        # draw and annotate on image
        #cv2.rectangle(image, (x, y), (xb, yb), (0, 0, 255), 2)  # RGB (0, 0, 255) or COLORS[detector_idxs[i]]  (255,0,255)
        cv2.rectangle(image, (x, y), (xb, yb), (0, 0, 255) , 2)
        cv2.putText(image, '{}:{}%'.format(modelname[detector_idxs[i]],str(percent)), (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[detector_idxs[i]] ,2 ) # RGB(128, 255, 0)
        print('{} - Detect {} logo with confidence {}%'.format(GetTimeString(), modelname[detector_idxs[i]] , percent ))
        if modelname[detector_idxs[i]] not in LogoList: LogoList.append(modelname[detector_idxs[i]] )

    stop = timeit.default_timer()
    print("GetMultidetectImage Spent Time: {}\n".format(stop - start))
    return image


def GetBarcodeDetectImage(image):
    global DUTSN, HWID, PN
    start = timeit.default_timer()

    decodedObjects = pyzbar.decode(image)

    bardata = ''
    for barcode in decodedObjects:
        barx = int(barcode.rect.left)
        bary = int(barcode.rect.top)
        barwidth = int(barcode.rect.width)
        barheight = int(barcode.rect.height)
        bardata = barcode.data.decode("utf-8")
        bartype =  barcode.type

        if bardata != '':
            # Get Osprey data when DUTSN change / Get Image Barcode data
            if len(bardata) == 16 and bardata.find('-') < 0 and DUTSN != bardata: ## when SN change
                DUTSN = bardata

            elif len(bardata) == 12 and bardata.find('-') > 0:
                PN = bardata

            elif len(bardata) == 16 and bardata.find('-') > 0:
                HWID = bardata

            # put image BarCodeData
            cv2.rectangle(image, (barx, bary), (barx + barwidth, bary + barheight), (0, 0, 255), 2)
            cv2.putText(image, str(bardata), (barx + 20, bary + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (204, 255, 255), 2)
            print('{} - Detect {} Barcode {}'.format(GetTimeString(), str(bardata), bardata))


    stop = timeit.default_timer()
    print("GetBarcodeDetectImage Spent Time: {}\n".format(stop - start))

    return image


def GetOCRDetectString(image):
    start = timeit.default_timer()

    # Image to String
    OCRresult = pytesseract.image_to_string(image , lang='eng' ) # , lang='eng' #, lang='eng+chi_tra' , config='--oem 2'

    # Remove Blank Space
    while '\n\n' in OCRresult:
        OCRresult = OCRresult.replace('\n\n', '\n')

    stop = timeit.default_timer()

    try :
        print(OCRresult)
    except :
        print( str(OCRresult).encode("utf8").decode("cp950", "ignore") )

    print ("GetOCRDetectString Spent Time: {}\n".format(stop - start))

    return OCRresult


def GetOCRDetectImage(image):

    #############################################################################################
    # Image Add OCR
    start = timeit.default_timer()

    boxes = pytesseract.image_to_boxes(image)
    print(str(boxes).encode("utf8").decode("cp950", "ignore"))
    OCRstring =''
    boxes = (boxes.split()) # string change to list

    try :
        h, w, _ = image.shape
    except:
        h, w = image.shape

    if int(w)>900 : fontscale = 0.6
    elif int(w)<900 : fontscale = 0.4

    for index, item in enumerate(boxes):
        OCRindex = index % 6
        if OCRindex == 0 :
            stringdata = item
            OCRstring += item
        elif OCRindex == 1 : barx = int(item)
        elif OCRindex == 2 : bary = int(item)
        elif OCRindex == 3 : barwidth = int(item)
        elif OCRindex == 4 : barheight = int(item)
        elif OCRindex == 5 :
            cv2.putText(image, str(stringdata), (barx, h- bary -20 ), cv2.FONT_HERSHEY_SIMPLEX, fontscale, (0, 255, 0), 1)


    stop = timeit.default_timer()

    print("GetOCRDetectImage Spent Time: {}\n".format(stop - start))


    return image , OCRstring




def GetTimeString():
    now = datetime.datetime.now()
    Timestring = '%s-%s-%s_%s-%s-%s' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return Timestring


def GetMultiBuildmodel():
    model = []
    model.append(dlib.fhog_object_detector('ML_model/CE.svm'))
    model.append(dlib.fhog_object_detector('ML_model/WEEE.svm'))
    model.append(dlib.fhog_object_detector('ML_model/ROHS.svm'))
    model.append(dlib.fhog_object_detector('ML_model/ULCSA.svm'))
    model.append(dlib.fhog_object_detector('ML_model/Fortinet.svm'))
    model.append(dlib.fhog_object_detector('ML_model/ICES_A.svm'))
    # model.append(dlib.fhog_object_detector('ML_model/ICES_B.svm'))
    model.append(dlib.fhog_object_detector('ML_model/RCM.svm'))
    model.append(dlib.fhog_object_detector('ML_model/Address.svm'))


    modelname = []
    modelname.append( os.path.splitext(os.path.basename('ML_model/CE.svm'))[0] )
    modelname.append( os.path.splitext(os.path.basename('ML_model/WEEE.svm'))[0] )
    modelname.append( os.path.splitext(os.path.basename('ML_model/ROHS.svm'))[0] )
    modelname.append( os.path.splitext(os.path.basename('ML_model/ULCSA.svm'))[0] )
    modelname.append( os.path.splitext(os.path.basename('ML_model/Fortinet.svm'))[0])
    modelname.append( os.path.splitext(os.path.basename('ML_model/ICES_A.svm'))[0] )
    # modelname.append( os.path.splitext(os.path.basename('ML_model/ICES_B.svm'))[0] )
    modelname.append( os.path.splitext(os.path.basename('ML_model/RCM.svm'))[0])
    modelname.append(os.path.splitext(os.path.basename('ML_model/Address.svm'))[0])

    global COLORS
    COLORS = np.random.uniform(0, 255, size=(len(model), 3))

    return  model,modelname

def resize(img):
    return cv2.resize(img, (640, 480))


def buttonbox_with_choice(string=None):
    global DUTSN, HWID, PN,LogoList
    global originalimg, newimg

    msg = "DUTSN={}\nLogoList={}".format(str(DUTSN),str(LogoList))
    if string !=None : msg+=string

    choices = ['Continue','SaveTesscrtOCRString','AddOCRImage','SaveFile','Reset_DUTSN']
    images = list()
    images.append(os.path.join("Original.jpg"))
    images.append(os.path.join("Result.jpg"))
    reply = easygui.buttonbox(msg, images=images, choices=choices)
    print("Reply was: {!r}".format(reply))

    if reply == 'SaveFile':
        shutil.copy("Original.jpg", str(DUTSN)+ '_'+ GetTimeString() + "_Original.jpg")
        shutil.copy("Result.jpg", str(DUTSN)+ '_'+ GetTimeString() + "_Result.jpg")

    elif reply == 'Reset_DUTSN':
        ResetGlobalVar()

    elif reply == 'SaveTesscrtOCRString':
        OCRresult = GetOCRDetectString(originalimg)
        with open(str(DUTSN)+ '_'+ GetTimeString() + "_OriginalOCR_tesseract.txt" , 'w',encoding="utf-8") as outfile:
            outfile.write(OCRresult)
        #Compare
        StringCompare= False
        StringCompareration = 0
        if 'Fortinet Inc' in OCRresult : StringCompare = True
        import difflib
        StringCompareration =  difflib.SequenceMatcher(None, 'Fortinet Inc', OCRresult).ratio()
        print ( 'Result={} Ratio={}'.format(str(StringCompare),StringCompareration) )
        buttonbox_with_choice('\nOCR Result:\n' + OCRresult)

    elif reply == 'AddOCRImage':
        OCRimg, OCRstr = GetOCRDetectImage(newimg) ## Tesscrt
        buttonbox_with_choice('\nOCR Result:\n' + OCRstr)

    elif reply == 'LogoImage':
        pass

    return reply



def ResetGlobalVar():
    global DUTSN, HWID, PN
    global LogoList
    LogoList = []
    DUTSN=None
    HWID=None
    PN=None

if __name__ == "__main__":
    global originalimg, newimg

    ResetGlobalVar()

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", default=None, help="please input your mobile IP...") #required=True
    ap.add_argument("-p", "--image", default=None, help="please input your image path...")  # required=True
    args = vars(ap.parse_args())

    models,modelname = GetMultiBuildmodel()

    print('Try Connection {}'.format(args["ip"]))
    url = 'http://{}:8080/shot.jpg'.format(args["ip"])

    BarCodeDetectEnabled=True
    CropImageEnabled = False
    LogoDetectEnabled=False
    OCRDetectEnabled=False
    RecordVideoEnabled=False
    imagetitle='SN'

    while True:
        # Use urllib to get the image and convert into a cv2 usable format
        try:
            import urllib.request
            imgResp = urllib.request.urlopen(url)  # python 3.5
        except:
            import urllib
            imgResp = urllib.urlopen(url)  # python 2.7

        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        originalimg = cv2.imdecode(imgNp, -1)


        if CropImageEnabled == True:
            cv2.imwrite("Original.jpg", originalimg)
            process_image("Original.jpg", "Original_Crop.jpg")
            newimg = cv2.imread("Original_Crop.jpg")
        else :
            newimg = originalimg


        #############################################################################################
        # Image Add LogoDetect
        if BarCodeDetectEnabled==True:
            newimg = GetBarcodeDetectImage(newimg)
        #############################################################################################

        #############################################################################################
        # Image Add LogoDetect
        if LogoDetectEnabled==True :
            newimg = GetMultidetectImage(models, modelname, newimg )
            time.sleep(0.1)
        #######################################

        #############################################################################################
        # Image Add OCR
        if OCRDetectEnabled==True :
            newimg , OCRstr = GetOCRDetectImage(newimg)
            OCRstr = OCRstr.encode(sys.stdin.encoding, "replace").decode(sys.stdin.encoding)  # workaround
            print ('GetOCRDetectImage={}'.format(OCRstr) )
            time.sleep(0.1)
        #######################################

        # put the image on screen
        #Method 1 Cv2
        cv2.imshow( imagetitle , newimg)

        # Title change need to destroy previous window
        if imagetitle != '{}_{}_{}_{}'.format(DUTSN,HWID,PN,str(LogoList)):
            deployimagetitle = imagetitle
            imagetitle = '{}_{}_{}_{}'.format(DUTSN,HWID,PN,str(LogoList))
            cv2.imshow(imagetitle, newimg)
            cv2.destroyWindow(deployimagetitle)

        # write the flipped frame
        if RecordVideoEnabled==True :
            out.write(newimg)

        # wait user keyword to action
        key = (cv2.waitKey(1))

        if key == 32: ##cv2.waitKey(1) & 0xFF == 32: ##space key ASCII= 32
            if LogoDetectEnabled == False :
                LogoDetectEnabled = True
                print ("[Enable LogoDetect]")
            else:
                LogoDetectEnabled = False
                print ("[Disable LogoDetect]")

        elif key == ord('o'):
            if OCRDetectEnabled == False :
                OCRDetectEnabled = True
                print ("[Enable OCRDetect]")
            else:
                OCRDetectEnabled = False
                print ("[Disable OCRDetect]")

        elif key == ord('b'):
            if BarCodeDetectEnabled == False :
                BarCodeDetectEnabled = True
                print ("[Enable BarCodeDetect]")
            else:
                BarCodeDetectEnabled = False
                print ("[Disable BarCodeDetect]")

        elif key == ord('\r'): ## cv2.waitKey(1)& 0xFF == ord('\r'): ## enter key ASCII ## cv2.waitKey(1)== 13
            cv2.imwrite("Original.jpg", originalimg)
            cv2.imwrite("Result.jpg", newimg)
            buttonbox_with_choice()

        elif key == ord('s'):
            time.sleep(10)

        elif key == ord('r'):
            if RecordVideoEnabled == False :
                RecordVideoEnabled = True
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                size = newimg.shape[1], newimg.shape[0]
                out = cv2.VideoWriter( GetTimeString()+ '_record.avi', fourcc, 20.0, size)
                print ( "[Start Record Video on {}]".format(GetTimeString()+ '_record.avi'))
            else:
                RecordVideoEnabled = False
                print ( "[End Record Video]" )

        elif key == ord('q'):  ## cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
