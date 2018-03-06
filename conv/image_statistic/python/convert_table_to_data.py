# -*- coding: utf-8 -*-


path = "/home/lyan/Documents/image_statistic/output/"
name = ["lentils1-a.png1.txt",
"sesameseeds1-d.png1.txt",
"ceiling2-c.png1.txt",
"sand1-d.png1.txt",
"floor1-a.png1.txt",
"blanket2-c.png1.txt",
"lentils1-d.png1.txt",
"blanket2-b.png1.txt",
"blanket2-d.png1.txt",
"cushion1-d.png1.txt",
"seat1-d.png1.txt",
"lentils1-b.png1.txt",
"blanket1-b.png1.txt",
"cushion1-a.png1.txt",
"floor1-b.png1.txt",
"blanket1-d.png1.txt",
"blanket2-a.png1.txt",
"ceiling2-d.png1.txt",
"blanket1-c.png1.txt",
"scarf1-b.png1.txt",
"cushion1-b.png1.txt",
"seat2-b.png1.txt",
"sesameseeds1-c.png1.txt",
"blanket1-a.png1.txt",
"lentils1-c.png1.txt",
"cushion1-c.png1.txt"]

size = 12

pat = ["Первый основной момент","Второй угловой момент", 
"Контраст", "Инерция", "Корреляция","Затенение","Энтропия", "Обратное отклонение","Обратный момент", "Диагональный момент", "Суммарное среднее", "Суммарная энтропия", "Суммарная корреляция"]


out = open('output.txt','w')

for n in name:
	out.write( "file:" + n + "\n")
	with  open(path+n) as f:
		lines = f.readlines()
		for i in range(0,size):
			out.write( (pat[i] +":"+ lines[i]).replace('\n','') + "\n")
	out.write("\n")
