def upload(video_path):
        """
        return example: {"fileIp":"10.50.200.107:8888",
        "fileUrl":"group1/M00/00/00/CjLIa10V-92AUONHAAAACV_xtOE8838105",
        "success":true}
        """
        import requests,json
        url='http://222.173.73.19:8762/mtrp/file/json/upload.jhtml'
        with open(video_path,'rb') as file:
            files = {'upload': file}
            r = requests.post(url, files=files)
        #print(r)
        result=json.loads(r.content)
        if (result['success']):
            return result['fileUrl']
        else:
            print('upload',video_path,'failed')
            print('file upload server return',r.content)
            return None

video_path = "/home/sy/keras-retinanet/wurenji/helmet/outVideo/out_200.mp4"
re = upload(video_path)
print(re)