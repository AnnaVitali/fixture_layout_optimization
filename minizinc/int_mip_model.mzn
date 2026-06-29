include "globals.mzn";

int: WTab = 2500;
int: HTab = 1500;
int: WMin = 200;
int: HMin = 100;
int: WBar = 145;

int: NTypes = 2;
array[1..NTypes] of int: _NFix = [24, 12];
int: NFix = sum(_NFix);
array[0..NTypes] of int: WFix = array1d(0..NTypes, [0, 145, 180]);
array[0..NTypes] of int: HFix = array1d(0..NTypes, [0, 145,  65]);

int: minFix;
int: maxFix;

int: M = WTab * HTab;

int: NExt;
array [1..NExt] of int: HExt;
array [1..NExt] of int: WExt;
array [1..NExt] of int: XExt;
array [1..NExt] of int: YExt;

int: NIneq;
array [1..NIneq, 1..5] of int: ineq_coeff;

array [1..maxFix] of var 0..WTab: x;
array [1..maxFix] of var 0..HTab: y;

var minFix..maxFix: n;

%MIP introduced variables
array [1..NFix, 1..NTypes] of var 0..1: T; % type of fixture
array [1..maxFix] of var 0..1: S; % fixture selected
array [1..maxFix] of var 0..maxFix: F;

array [1..maxFix] of var 0..WTab: xc;
array [1..maxFix] of var 0..HTab: yc;
array [1..maxFix] of var 0..max(WFix): wx;
array [1..maxFix] of var 0..max(HFix): hx;

array [1..maxFix, 1..maxFix] of var 0..1: same_bar;
array [1..maxFix, 1..maxFix] of var 0..1: below_in_bar;
array [1..maxFix, 1..maxFix] of var 0..1: above_in_bar;

array [1..maxFix, 1..maxFix] of var 0..1: zij_above;
array [1..maxFix, 1..maxFix] of var 0..1: zij_below;
array [1..maxFix, 1..maxFix] of var 0..1: zij_left;
array [1..maxFix, 1..maxFix] of var 0..1: zij_right;
array [1..maxFix, 1..maxFix] of var 0..1: same_type;

array [1..maxFix, 1..maxFix] of var 0..1: ext_below;
array [1..maxFix, 1..maxFix] of var 0..1: ext_above;
array [1..maxFix, 1..maxFix] of var 0..1: ext_left;
array [1..maxFix, 1..maxFix] of var 0..1: ext_right;

array [1..maxFix, 1..maxFix] of var 0..WTab: xc_diff;
array [1..maxFix, 1..maxFix] of var 0..HTab: yc_diff;
array [1..maxFix, 1..maxFix] of var 0..1: xc_diff_positive;
array [1..maxFix, 1..maxFix] of var 0..1: yc_diff_positive;


constraint n = sum(S);

constraint forall(j in 1..NTypes)(
  sum([T[i, j] | i in 1..maxFix]) <= _NFix[j]
);

constraint forall(i in 1..maxFix)(
  S[i] = sum([T[i, j] | j in 1..NTypes])
);

constraint sum([T[i, j] | i in 1..maxFix, j in 1..NTypes]) <= maxFix /\ sum([T[i, j] | i in 1..maxFix, j in 1..NTypes]) >= minFix;

constraint forall(i in 1..maxFix)(
  wx[i] = sum([T[i,j] * WFix[j] | j in 1..NTypes]) /\ hx[i] = sum([T[i,j] * HFix[j] | j in 1..NTypes])
);

constraint forall(i in 1..maxFix)(
  wx[i] <= M * S[i] /\ hx[i] <= M * S[i]
);

constraint forall(i in 1..maxFix)(
  xc[i] = x[i] + wx[i] div 2 /\ yc[i] = y[i] + hx[i] div 2
);

%respect horizontal distance
constraint forall(i in 1..maxFix, j in (i + 1)..maxFix)(
  same_bar[i, j] <= S[i] /\
  same_bar[i, j] <= S[j] /\
  same_bar[i, j] = same_bar[j, i] /\ 
  xc[i] - xc[j] - 1 <= M * (1 - same_bar[i, j]) /\
  xc[j] - xc[i] - 1 <= M * (1 - same_bar[i, j]) /\
  xc[i] + WBar + WMin <= xc[j] + M * (same_bar[i,j] + zij_right[i, j]) /\
  xc[j] + WBar + WMin <= xc[i] + M * (same_bar[i,j] + zij_left[i, j])
);

% respect vertical distance
constraint forall(i in 1..maxFix, j in (i + 1)..maxFix)(
  below_in_bar[i, j] + above_in_bar[i, j] = same_bar[i, j] /\
  y[i] + hx[i] + HMin - y[j] <= M * (1 - zij_below[i, j]) /\
  y[j] + hx[j] + HMin - y[i] <= M * (1 - zij_above[i,j])
);

% fixture no overlap
constraint forall(i in 1..maxFix, j in 1..maxFix where i < j)(
  zij_above[i, j] + zij_right[i, j] + zij_below[i, j] + zij_left[i, j] >= 1 /\
  x[i] + wx[i] <= x[j] + M * (1 - zij_left[i,j]) /\
  x[j] + wx[j] <= x[i] + M * (1 - zij_right[i,j]) /\
  y[i] + hx[i] <= y[j] + M * (1 - zij_below[i,j]) /\
  y[j] + hx[j] <= y[i] + M * (1 - zij_above[i,j])
);

% no overlap with extruded areas
constraint forall(i in 1..maxFix, j in 1..NExt)(
  ext_above[i, j] + ext_right[i, j] + ext_below[i, j] + ext_left[i, j] >= S[i] /\
  x[i] + wx[i] <= XExt[j] + M * (1 - ext_left[i,j]) /\
  XExt[j] + WExt[j] <= x[i] + M * (1 - ext_right[i,j]) /\
  y[i] + hx[i] <= YExt[j] + M * (1 - ext_below[i,j]) /\
  YExt[j] + HExt[j] <= y[i] + M * (1 - ext_above[i,j]) 
);

% symmetry breaking
constraint forall(i in 1..maxFix)(
  F[i] = sum([j * T[i, j] | j in 1..NTypes])
);

constraint forall(i in 1..maxFix - 1)(
  F[i] >= F[i + 1]
);

constraint forall (j in 1..NIneq, i in 1..n)(
let {
  var int: x1 = x[i] + wx[i];
  var int: y1 = y[i] + hx[i];
} in
  ineq_coeff[j, 1] * x[i] + ineq_coeff[j, 2] * y[i] + ineq_coeff[j, 3] * x1 + ineq_coeff[j, 4] * y1 + ineq_coeff[j, 5] <= M * (1 - S[i])
);

constraint forall(i, j in 1..maxFix where i < j)(
  xc[i] - xc[j] <= M * xc_diff_positive[i, j] /\
  xc[j] - xc[i] <= M * (1 - xc_diff_positive[i, j]) /\
  yc[i] - yc[j] <= M * yc_diff_positive[i, j] /\
  yc[j] - yc[i] <= M * (1 - yc_diff_positive[i, j])
);

constraint forall(i, j in 1..maxFix where i < j)(
  xc_diff[i, j] >=  xc[i] - xc[j] /\ 
  xc_diff[i, j] >=  xc[j] - xc[i] /\
  xc_diff[i, j] <= (xc[i] - xc[j]) + M * (1 - xc_diff_positive[i, j]) /\
  xc_diff[i, j] <= (xc[j] - xc[i]) + M * xc_diff_positive[i, j] /\
  yc_diff[i, j] >=  yc[i] - yc[j] /\ 
  yc_diff[i, j] >=  yc[j] - yc[i] /\
  yc_diff[i, j] <= (yc[i] - yc[j]) + M * (1 - yc_diff_positive[i, j]) /\
  yc_diff[i, j] <= (yc[j] - yc[i]) + M * yc_diff_positive[i, j]
);

solve maximize sum(i, j in 1..maxFix where i < j)(
    (xc_diff[i, j] + yc_diff[i, j]) * S[i] * S[j]
    )
    + sum(i in 1..maxFix)(
      (wx[i] * hx[i]) * S[i]
    )
  ;

output [
  "{\n",
  "\"x\": ", show([x[i] | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"y\": ", show([y[i] | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"x1\": ", show([x[i] | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"y1\": ", show([(y[i] + hx[i]) | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"x2\": ", show([(x[i] + wx[i]) | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"y2\": ", show([(y[i] + hx[i]) | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"x3\": ", show([(x[i] + wx[i]) | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"y3\": ", show([y[i] | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"angle\": ", show([0 | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"fixtures_center_x\": ", show([xc[i] | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"fixtures_center_y\": ", show([yc[i] | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"bars_center\": ", show([x[i] + (wx[i] / 2) | i in 1..maxFix where S[i] = 1]), ",\n",
  "\"selected_fixture\": ", show([1 | i in 1..maxFix, j in 1..NTypes where T[i, j] == 1]), ",\n",
  "\"fixture_type\": ", show([j * T[i, j] | i in 1..maxFix, j in 1..NTypes where T[i, j] == 1]), ",\n",
  "\"objective_value\": ", show(_objective), "\n",
  "}\n"
];

