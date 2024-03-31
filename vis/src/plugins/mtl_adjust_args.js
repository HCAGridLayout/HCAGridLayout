class MTLAdjustArgs {
    constructor(...losses) {
        // losses: different loss functions
        this.losses = losses;
        this.T = 0;
        this.K = losses.length;
        this.lossHistory = [losses.map(() => 0)];
        this.argsHistory = [losses.map(() => 1)];
    }

    getloss(...inputs) {
        return inputs.map((input, i) => this.losses[i](...input));
    }

    getvalueNxt(lossValues) {
        return lossValues.reduce((sum, loss, i) => sum + loss * this.argsHistory[this.T][i], 0);
    }

    getvalueCur(lossValues) {
        return lossValues.reduce((sum, loss, i) => sum + loss * this.argsHistory[this.T - 1][i], 0);
    }

    adjust(lossValues, show = false) {
        // inputs: inputs to loss functions
        let t0Values = this.lossHistory[this.T];
        this.T += 1;

        let t1Values = lossValues;
        let weight1 = t1Values.map((t1, i) => {
            if (t1 === 0) return 1;
            let t0 = t0Values[i];
            let w = t0 === 0 ? this.T : t1 / t0;
            return Math.exp(w / this.T);
        });
        let weightSum = weight1.reduce((a, b) => a + b);
        let args1 = weight1.map((w) => w / weightSum * this.K);
        this.lossHistory.push(t1Values);
        // console.log('adjust', lossValues, t1Values, args1);
        this.argsHistory.push(args1);

        let minIndex = 0;
        let min = weight1[0];
        weight1.forEach((w, i) => {
            if (w < min) {
                minIndex = i;
                min = w;
            }
        });
        // if (show) console.log('adjust', weight1, this);
        return minIndex;
    }
}

export default MTLAdjustArgs;
