function range(start, stop, step) {
    if (step === undefined) step = 1;
    var a = [], b = start;
    while (b < stop) {
        a.push(b);
        b += step;
    }
    return a;
}

export default range;