var RBM;
(function(){
    // Check if running in Web Worker
    var inWorker = (self.constructor.toString().indexOf("Worker") !== -1);

    RBM = function(num_visible, num_hidden, learning_rate){
        this.num_hidden = num_hidden;
        this.num_visible = num_visible;
        this.learning_rate = learning_rate || 0.1;

    // Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
    // a Gaussian distribution with mean 0 and standard deviation 0.1.
        this.weights = each(function(x){return x*0.1}, randn(num_visible, num_hidden));

    // Insert weights for the bias units into the first row and first column.
        this.weights.unshift([]);
        for(var i = 0; i < num_hidden; i++) this.weights[0][i] = 0;
        for(i = 0; i <= num_visible; i++) this.weights[i].unshift(0);
    };

    /**
     * Train the machine.
     *
     * @param data A matrix where each row is a training example consisting of the states of visible units.
     * @param max_epochs
     */
    RBM.prototype.train = function(data, max_epochs){
        data = JSON.parse(JSON.stringify(data));   // Don't edit original array
        var self = this, error,
            num_examples = data.length,
            pos_hidden_activations, pos_hidden_probs, pos_hidden_states,
            neg_hidden_activations, neg_hidden_probs,
            neg_visible_activations, neg_visible_probs,
            pos_associations, neg_associations;
        max_epochs = max_epochs || 1000;

    // Insert bias units of 1 into the first column.
        for(var i = 0; i < data.length; i++) data[i].unshift(1);

        for(var epoch = 0; epoch < max_epochs; epoch++){
        // Clamp to the data and sample from the hidden units.
        // (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = dot(data, self.weights);
            pos_hidden_probs = each(self._logistic, pos_hidden_activations);
            pos_hidden_states = each(
                function(a,b){return a > b},
                pos_hidden_probs,
                rand(num_examples, self.num_hidden + 1)
            );
        // Note that we're using the activation *probabilities* of the hidden states, not the hidden states
        // themselves, when computing associations. We could also use the states; see section 3 of Hinton's
        // "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = dot(transpose(data), pos_hidden_probs);

        // Reconstruct the visible units and sample again from the hidden units.
        // (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = dot(pos_hidden_states, transpose(self.weights));
            neg_visible_probs = each(self._logistic, neg_visible_activations);
            for(i = 0; i < neg_visible_probs.length; i++) neg_visible_probs[i][0] = 1; // Fix the bias unit.
            neg_hidden_activations = dot(neg_visible_probs, self.weights);
            neg_hidden_probs = each(self._logistic, neg_hidden_activations);
        // Note, again, that we're using the activation *probabilities* when computing associations,
        // not the states themselves.
            neg_associations = dot(transpose(neg_visible_probs), neg_hidden_probs);

        // Update weights.
            self.weights = each(function(p, n, w){
                return w + self.learning_rate * ((p - n) / num_examples);
            }, pos_associations, neg_associations, self.weights);

            error = sum(each(function(a, b){ return (a - b) * (a - b) }, data, neg_visible_probs));
            if(inWorker){
                SELF.postMessage({
                    progress_update: (100*(epoch+1)/max_epochs) + "%",
                    error: error
                })
            } else {
                console.log("Epoch %s: error is %s", epoch, error);
            }
        }
        if(inWorker){
            SELF.postMessage({
                passback: worker_passback,
                result: self.weights
            })
        }
    };

    /**
     * Assuming the RBM has been trained (so that weights for the network have been learned),
     * run the network on a set of visible units, to get a sample of the hidden units.
     *
     * @param data A matrix where each row consists of the states of the visible units.
     * @return hidden_states A matrix where each row consists of the hidden units activated from the visible
     */
    RBM.prototype.run_visible = function(data){
        data = JSON.parse(JSON.stringify(data));   // Don't edit original array
        var self = this,
            num_examples = data.length,
            hidden_states, hidden_activations, hidden_probs;

    // Create a matrix, where each row is to be the hidden units (plus a bias unit)
    // sampled from a training example.
        hidden_states = ones(num_examples, self.num_hidden + 1);    // todo is this necessary?

    // Insert bias units of 1 into the first column of data.
        for(var i = 0; i < data.length; i++) data[i].unshift(1);

    // Calculate the activations of the hidden units.
        hidden_activations = dot(data, self.weights);
    // Calculate the probabilities of turning the hidden units on.
        hidden_probs = each(self._logistic, hidden_activations);
    // Turn the hidden units on with their specified probabilities.
        hidden_states = each(
            function(a,b){ return a > b ? 1 : 0 },
            hidden_probs,
            rand(num_examples, self.num_hidden + 1)
        );
    // Always fix the bias unit to 1.
    // hidden_states[:,0] = 1

    // Ignore the bias units.
        for(i = 0; i < hidden_states.length; i++) hidden_states[i].shift();

        if(inWorker){
            SELF.postMessage({
                passback: worker_passback,
                result: hidden_states
            })
        }

        return hidden_states
    };
    /**
     * Assuming the RBM has been trained (so that weights for the network have been learned),
     * run the network on a set of hidden units, to get a sample of the visible units.
     *
     * @param data A matrix where each row consists of the states of the hidden units.
     * @return visible_states: A matrix where each row consists of the visible units activated from the hidden
     * units in the data matrix passed in.
     */
    RBM.prototype.run_hidden = function(data){
        data = JSON.parse(JSON.stringify(data));   // Don't edit original array
        var self = this,
            num_examples = data.length,
            visible_states, visible_activations, visible_probs;

    // Create a matrix, where each row is to be the visible units (plus a bias unit)
    // sampled from a training example.
        visible_states = ones(num_examples, self.num_visible + 1);

    // Insert bias units of 1 into the first column of data.
        for(var i = 0; i < data.length; i++) data[i].unshift(1);

    // Calculate the activations of the visible units.
        visible_activations = dot(data, transpose(self.weights));
    // Calculate the probabilities of turning the visible units on.
        visible_probs = each(self._logistic, visible_activations);
    // Turn the visible units on with their specified probabilities.
        visible_states = each(
            function(a,b){ return a > b ? 1 : 0 },
            visible_probs,
            rand(num_examples, self.num_visible + 1)
        );
    // Always fix the bias unit to 1.
    // visible_states[:,0] = 1

    // Ignore the bias units.
        for(i = 0; i < visible_states.length; i++) visible_states[i].shift();

        if(inWorker){
            SELF.postMessage({
                passback: worker_passback,
                result: visible_states
            })
        }

        return visible_states
    };

    /**
     * Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
     * (where each step consists of updating all the hidden units, and then updating all of the visible units),
     * taking a sample of the visible units at each step.
     * Note that we only initialize the network *once*, so these samples are correlated.
     *
     * @param num_samples
     * @return samples: A matrix, where each row is a sample of the visible units produced while the network was
     * daydreaming.
     */
    RBM.prototype.daydream = function(num_samples){
        var self = this,
            samples, visible,
            hidden_activations, hidden_probs, hidden_states,
            visible_activations, visible_probs, visible_states;

    // Create a matrix, where each row is to be a sample of of the visible units
    // (with an extra bias unit), initialized to all ones.
        samples = ones(num_samples, self.num_visible + 1);

    // Take the first sample from a uniform distribution.
        for(var i = 1; i < samples[0].length; i++){
            samples[0][i] = rand();
        }

    // Start the alternating Gibbs sampling.
    // Note that we keep the hidden units binary states, but leave the
    // visible units as real probabilities. See section 3 of Hinton's
    // "A Practical Guide to Training Restricted Boltzmann Machines"
    // for more on why.
        for(i = 0; i < num_samples-1; i++){
            visible = samples[i];

        // Calculate the activations of the hidden units.
            hidden_activations = dot(visible, self.weights)[0];
        // Calculate the probabilities of turning the hidden units on.
            hidden_probs = each(self._logistic, hidden_activations);
        // Turn the hidden units on with their specified probabilities.
            hidden_states = each(function(a, b){ return a > b }, hidden_probs, rand(self.num_hidden + 1));
        // Always fix the bias unit to 1.
            hidden_states[0] = true;

        // Recalculate the probabilities that the visible units are on.
            visible_activations = dot(hidden_states, transpose(self.weights))[0];
            visible_probs = each(self._logistic, visible_activations);
            visible_states = each(function(a, b){ return a > b ? 1 : 0 }, visible_probs, rand(self.num_visible + 1));
            samples[i+1] = visible_states
        }
    // Ignore the bias units (the first column), since they're always set to 1.
        for(i = 0; i < num_samples; i++) samples[i].shift();

        if(inWorker){
            SELF.postMessage({
                passback: worker_passback,
                result: samples
            })
        }

        return samples;
    };


    RBM.prototype._logistic = function(x){
        return 1.0 / (1 + Math.exp(-x));
    };

// Some helpful functions
    var dot = function(a, b){
            var ah = a.length, bh = b.length,
                aw = a[0].length, bw = b[0].length;

            if(typeof aw === "undefined"){
                a = each(function(x){return [x]}, a);
                aw = 1;
            }
            if(typeof bw === "undefined"){
                b = each(function(x){return [x]}, a);
                bw = 1;
            }

            if (aw != bh) {
                if(ah === bh){   // try multiplying flipped matrix
                    a = transpose(a);
                    aw = ah;
                    ah = a.length;
                } else {
                    throw Error("incompatible sizes");
                }
            }

            var result = [];
            for(var i = 0; i < ah; i++){
                result[i] = [];
                for(var j = 0; j < bw; j++){
                    var sum = 0;
                    for(var k = 0; k < aw; k++){
                        sum += a[i][k] * b[k][j];
                    }
                    result[i][j] = sum;
                }
            }
            return result;
        },
        transpose = function(a){
            var w = a.length ? a.length : 0,
                h = a[0] instanceof Array ? a[0].length : 0;
            if(h === 0 || w === 0) { return []; }
            for(var i = 0, t = [], j; i < h; i++) {
                t[i] = [];
                for(j = 0; j < w; j++) {
                    t[i][j] = a[j][i];
                }
            }
            return t;
        },
        ones = function(){
            if(typeof arguments[0] === "undefined") return 1;
            for(var i = 0, arr = []; i < arguments[0]; i++){
                arr.push(ones.apply(this, Array.prototype.slice.call(arguments, 1)));
            }
            return arr;
        },
        randn = function(){ // Box-Muller - too slow?
            if(typeof arguments[0] === "undefined") return Math.sqrt(-2*Math.log(Math.random()))*Math.cos(2*Math.PI*Math.random());
            for(var i = 0, arr = []; i < arguments[0]; i++){
                arr.push(randn.apply(this, Array.prototype.slice.call(arguments, 1)));
            }
            return arr;
        },
        rand = function(){
            if(typeof arguments[0] === "undefined") return Math.random();
            for(var i = 0, arr = []; i < arguments[0]; i++){
                arr.push(rand.apply(this, Array.prototype.slice.call(arguments, 1)));
            }
            return arr;
        },
        sum = function(a){
            if(!(a instanceof Array)) return a;
            for(var i = 0, s = 0, l = a.length; i < l; i++) s += sum(a[i]);
            return s;
        },
        each = function(operation){
            if(arguments.length <= 1) return null;
            var args = Array.prototype.slice.call(arguments, 1);

        // check if we have any arrays or undefined arguments
            for(var i = 0, arrays = false; i < args.length; i++){
                if(typeof args[i] === "undefined") return null;
                if(!arrays && args[i] instanceof Array) arrays = true;
            }

            if(!arrays) return operation.apply(this, args); // can run function if no arrays

        // get max length of argument arrays so we don't return something too short
        // (we can fill with nulls if the lengths don't match)
            var max_length = 0;
            for(i = 0; i < args.length; i++){
                if(args[i] instanceof Array){
                    if(args[i].length > max_length) max_length = args[i].length
                } else if(1 > max_length) max_length = 1
            }

        // recursion baby!
            var arr = [];
            for(i = 0; i < max_length; i++){
                for(var j = 0, new_args = [operation]; j < args.length; j++) new_args.push(args[j][i]);
                arr.push(each.apply(this, new_args))
            }
            return arr;
        }

    // WebWorker stuff
    // Recieves a message containing an object of the form {
    //      rbm : /* The RBM object */
    //      cmd : /* The method to call (eg. train, classify) */       
    //      args : /* An array of arguments to pass to the function */
    // }
    // While running, sends back messages containing an object of the form {
    //      progress_update : /* The percentage completion */
    // }
    // On completion, sends back a message containing an object of the form {
    //      passback : /* The request object */
    //      result : /* The result of the function */
    // }
    if(inWorker){
        var SELF = self,
            worker_passback;
        SELF.addEventListener("message", function(e) {
            var data = e.data || {};
            if(data.hasOwnProperty("cmd") && RBM.prototype.hasOwnProperty(data["cmd"]) && typeof RBM.prototype[data["cmd"]] === "function"){
                var rbm = Object.create(RBM.prototype); // Reconstruct RBM
                for(var prop in data["rbm"]) rbm[prop] = data["rbm"][prop];
                worker_passback = data;    // For passing the request back
                RBM.prototype[data["cmd"]].apply(rbm, data["args"]);   // Run command
            }
        }, false);
    }
})();
