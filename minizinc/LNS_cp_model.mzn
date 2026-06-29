include "globals.mzn";

int: WTab = 2500;
int: HTab = 1500;
int: WMin = 200;
int: HMin = 100;
int: NTypes = 2;
int: WBar = 145;
array[1..NTypes] of int: _NFix = [24, 12];
array[0..NTypes] of int: WFix = array1d(0..NTypes, [0, 145, 180]);
array[0..NTypes] of int: HFix = array1d(0..NTypes, [0, 145,  65]);

int: minFix; % min no. of fixtures.
int: maxFix; % max no. of fixtures.
int: maxFixBars = 3;

array [1..maxFix] of var 0..WTab: x;
array [1..maxFix] of var 0..HTab: y;
array [1..maxFix] of var 0..NTypes: t;

int: NExt;
array [1..NExt] of int: HExt;
array [1..NExt] of int: WExt;
array [1..NExt] of int: XExt;
array [1..NExt] of int: YExt;

int: NIneq;
array [1..NIneq, 1..5] of int: ineq_coeff;

array [1..NTypes] of var 0..maxFix: _n;
var minFix..maxFix: n;

function var int:
xc(var 0..maxFix: i) ::promise_total = (x[i] + (WFix[t[i]] div 2));
function var int:
yc(var 0..maxFix: i) ::promise_total = (y[i] + (HFix[t[i]] div 2));

constraint forall(i in 1..maxFix) (
let {
  var {0, 144, 180}: wx = WFix[t[i]] - (WFix[t[i]] mod 2);
  var {0, 144,  64}: wy = HFix[t[i]] - (HFix[t[i]] mod 2);
} in
  2*xc(i) - 2*x[i] = wx /\
  2*yc(i) - 2*y[i] = wy /\
  wx >= wy
);

constraint n = sum(_n);

constraint _n = [count(t,i) | i in 1..NTypes] /\ count(t,0) = maxFix - n;

constraint forall(i in 1..NTypes)(_n[i] <= _NFix[i]);

constraint lex_chain(array2d(1..2, 1..maxFix, x ++ y));

constraint forall (i in 1..maxFix) (
  (i > n) <-> (x[i] = WTab /\ y[i] = HTab /\ t[i] = 0)
);

constraint diffn_nonstrict(
  x ++ XExt, y ++ YExt, [WFix[t[i]] | i in 1..maxFix] ++ WExt, [HFix[t[i]] | i in 1..maxFix] ++ HExt
);

constraint forall (i, j in 1..n where i < j) (
 [xc(i) + WBar + WMin <= xc(j), y[i] + HFix[t[i]] + HMin <= y[j] \/ y[j] + HFix[t[j]] + HMin  <= y[i]][(xc(i) == xc(j)) + 1]
 );

constraint forall (i in 2..n where xc(i-1) < xc(i)) (
  xc(i-1) + WBar + WMin <= xc(i)
);

constraint forall (j in 1..NIneq, i in 1..n)(
let {
  var int: x1 = x[i] + WFix[t[i]];
  var int: y1 = y[i] + HFix[t[i]];
} in
  ineq_coeff[j, 1] * x[i] + ineq_coeff[j, 2] * y[i] + ineq_coeff[j, 3] * x1 + ineq_coeff[j, 4] * y1 + ineq_coeff[j, 5] <= 0
);

% Maximize distance between pairs of centroids plus sum widths + heights
solve
  :: restart_constant(1000)
  :: relax_and_reconstruct(x ++ y, 60)
 maximize sum(i, j in 1..n where i < j)(
abs(xc(i) - xc(j)) + abs(yc(i) - yc(j))
) + sum(i in 1..n)(WFix[t[i]] * HFix[t[i]]);

output [
  "{\n",
  "\"x\": ", show([x[i] | i in 1..n]), ",\n",
  "\"y\": ", show([y[i] | i in 1..n]), ",\n",
  "\"x1\": ", show([x[i] | i in 1..n]), ",\n",
  "\"y1\": ", show([(y[i] + HFix[t[i]]) | i in 1..n]), ",\n",
  "\"x2\": ", show([(x[i] + WFix[t[i]]) | i in 1..n]), ",\n",
  "\"y2\": ", show([(y[i] + HFix[t[i]]) | i in 1..n]), ",\n",
  "\"x3\": ", show([(x[i] + WFix[t[i]]) | i in 1..n]), ",\n",
  "\"y3\": ", show([y[i] | i in 1..n]), ",\n",
  "\"angle\": ", show([0 | i in 1..n]), ",\n",
  "\"fixtures_center_x\": ", show([(x[i] + (WFix[t[i]] / 2)) | i in 1..n]), ",\n",
  "\"fixtures_center_y\": ", show([(y[i] + (HFix[t[i]] / 2)) | i in 1..n]), ",\n",
  "\"bars_center\": ", show([x[i] + (WFix[t[i]] div 2) | i in 1..n]), ",\n",
  "\"selected_fixture\": ", show([1 * (t[i] != 0) | i in 1..n]), ",\n",
  "\"fixture_type\": ", show([t[i] | i in 1..n]), ",\n",
  "\"objective_value\": ", show(_objective), "\n",
  "}\n"
];
