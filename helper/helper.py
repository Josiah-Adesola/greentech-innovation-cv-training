import yaml
data = yaml.safe_load(open("data.yaml"))
print("Class names:", data["names"])
print("Total classes (nc):", len(data["names"]))
