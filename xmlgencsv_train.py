import string
import sys,os
try:
  import xml.etree.cElementTree as ET
except ImportError:
  import xml.etree.ElementTree as ET

def getAllXml(path):
    txt = []
    cls = []
    for root, dirs, files in os.walk(path):
        for img_name in files:
            if not img_name.lower ().endswith (('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            imgpath = os.path.join (root, img_name)
            xml_name = img_name[0:img_name.find (".")]+".xml"
            xmlpath = os.path.join (root, xml_name)
            if (os.path.isfile (xmlpath) == False): continue
            tree = ET.parse (xmlpath)
            if (tree == None): continue
            gen = tree.getroot ()
            obj = gen.findall ("object")
            for item in obj:
                str = ""
                str += imgpath
                name = item.find ("name").text
                if (name.find ("snail") != -1):
                    name = "snail"
                elif (name.find ("coin")!=-1):
                    name="coin"
                elif (name.find ("eel") != -1):
                    name = "eel"
                elif (name.find ("female_sturgeon") != -1):
                    name = "female_sturgeon"
                elif (name.find ("male_sturgeon") != -1):
                    name = "male_sturgeon"
                elif(name.find("female")!=-1):
                    name="female"
                elif (name.find ("male") != -1):
                     name = "male"
                elif (name.find ("carapace") != -1):
                    name = "carapace"
                elif (name.find ("carb") != -1 or name.find ("crab") != -1):
                    name = "crab"
                else:
                    print("label error!,name is :",name)
                    continue
                bndbox = item.find ("bndbox")
                x1 = bndbox.find ("xmin").text
                y1 = bndbox.find ("ymin").text
                x2 = bndbox.find ("xmax").text
                y2 = bndbox.find ("ymax").text
                str += "," + (x1) + "," + (y1) + "," + (x2) + "," + (y2)
                #str += ",crab"
                str +=","+name
                if(name not in cls):
                    cls.append(name)
                txt.append (str)
    return txt,cls


if __name__ == '__main__':
    path="/home/sy/crab/used/snail/"
    txt,cls=getAllXml(path)
    title="iscas_frcnn_train"
    cnt = 1
    with open ("%s.csv" % title, "w") as f:
        for item in txt:
            cnt +=1
            print (item)
            f.write (item)
            f.write ("\n")
    f.close()
    print("count:",cnt)

    title = "iscas_frcnn_class_train"
    id = 0
    with open ("%s.csv" % title, "w") as f:
        for item in cls:
            print (item)
            item = item +",%d"%id
            id += 1
            f.write (item)
            f.write ("\n")
    f.close()

