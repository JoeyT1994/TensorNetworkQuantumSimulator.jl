// Every one- and two-qubit qiskit standard gate, written by qiskit.qasm2.dumps (qiskit 2.2.3).
OPENQASM 2.0;
include "qelib1.inc";
gate cs q0,q1 { t q0; cx q0,q1; tdg q1; cx q0,q1; t q1; }
gate csdg q0,q1 { tdg q0; cx q0,q1; t q1; cx q0,q1; tdg q1; }
gate dcx q0,q1 { cx q0,q1; cx q1,q0; }
gate ecr q0,q1 { s q0; sx q1; cx q0,q1; x q0; }
gate iswap q0,q1 { s q0; s q1; h q0; cx q0,q1; cx q1,q0; h q1; }
gate r(param0,param1) q0 { u(param0,-pi/2 + param1,pi/2 - param1) q0; }
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate xx_minus_yy(param0,param1) q0,q1 { rz(-param1) q1; sdg q0; sx q0; s q0; s q1; cx q0,q1; ry(0.5*param0) q0; ry((-0.5)*param0) q1; cx q0,q1; sdg q1; sdg q0; sxdg q0; s q0; rz(param1) q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
qreg q[4];
ry(-2.333762244609258) q[0];
ry(-0.004537324106032248) q[1];
ry(0.6377329893219388) q[2];
ry(-2.961334297709639) q[3];
ch q[0],q[1];
cp(-2.212145652424117) q[2],q[1];
crx(2.690529207836934) q[2],q[3];
cry(-2.6991271241746224) q[1],q[0];
crz(-2.326198881469457) q[3],q[2];
cs q[1],q[2];
csdg q[0],q[1];
csx q[2],q[1];
cu(2.8169307505134302,0.7658171994444922,-0.8231404801204714,0.07156561764271485) q[2],q[3];
cu1(1.0231724466312961) q[1],q[0];
cu3(-1.4117763474624538,-2.274713685292257,1.8098061480734478) q[3],q[2];
cx q[1],q[2];
cy q[0],q[1];
cz q[2],q[1];
dcx q[2],q[3];
ecr q[1],q[0];
h q[0];
id q[1];
iswap q[0],q[1];
p(1.0704071189552584) q[3];
r(0.07780037014628549,1.9901137207329835) q[0];
rx(0.3083490083100373) q[1];
rxx(3.0216695124550927) q[3],q[2];
ry(-1.8566218109816433) q[3];
ryy(0.337597826504227) q[0],q[1];
rz(-0.1028890636915829) q[1];
rzx(-0.9219012748779876) q[2],q[3];
rzz(0.5755102679792783) q[1],q[0];
s q[0];
sdg q[1];
swap q[0],q[1];
sx q[3];
sxdg q[0];
t q[1];
tdg q[2];
u(-1.663151412014791,1.8987954625072057,2.3080247757703507) q[3];
u1(-2.3325717791736853) q[0];
u2(-0.20688514363463462,-1.4002399368621565) q[1];
u3(-2.6193531546428392,2.4877914600600946,-0.44014534887375323) q[2];
x q[3];
xx_minus_yy(-2.2136208476697106,1.0892678160540203) q[3],q[2];
xx_plus_yy(-1.871031878915773,2.522265855458806) q[1],q[2];
y q[2];
z q[3];
