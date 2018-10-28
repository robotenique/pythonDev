const millis = () => Date.now();
const flip_coin = () => {
    let n = 0
    const then = millis() + 1;
    while (millis() <= then) {
        n = !n
    }
    return n
}

const get_fair_bit = () => {
    while (1) {
        let a = flip_coin()
        if (a != flip_coin()) return a
    }
}

const get_random_byte = () => {
    let n = 0, bits = 8
    while(bits--) {
        n <<= 1
        n |= get_fair_bit()
    }
    return n
}

for (let i = 0; i < 1e5; i++) {
    let content = document.createTextNode(get_random_byte().toString() + "\n")
    document.getElementById("target").appendChild(content)
}