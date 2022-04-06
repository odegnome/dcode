from dm_control import mjcf

model = mjcf.from_path('quad.xml')

print(model.find('geom', 'quadbody').get_attributes())
print(model.find('body', 'quad').get_attributes())