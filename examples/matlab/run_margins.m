opspec = operspec('swingup_rbf');
op = findop('swingup_rbf',opspec);
io(1) = linio('swingup_rbf/Input',1,'input');
io(2) = linio('swingup_rbf/Gain',1,'output');
sys = linearize('swingup_rbf',op,io);
allmargin(sys)
opspec = operspec('swingup_linear');
op = findop('swingup_linear',opspec);
io(1) = linio('swingup_linear/Input',1,'input');
io(2) = linio('swingup_linear/Gain',1,'output');
sys = linearize('swingup_linear',op,io);
allmargin(sys)