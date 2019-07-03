import requests
 
#请求地址
p = {}
p["app_id"]="1106571733"
p["time_stamp"]="lalal"
p["model"]="住嘉佳园"	

#url = "https://crg.wiseom.cn/helmet?id=1234"
#发送get请求
r = requests.post("https://crg.wiseom.cn/helmet",data=p)
#获取返回的json数据 
print(r.json(),type(r),type(r.json()))
