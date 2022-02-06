use rand::Rng;

/**
 * The RandPoll struct provide s sequence of indices to be iterated randomly without repeating an element efficiently.
 * Thread UNSAFE.
 */
pub struct RandPoll {
    v: Vec<usize>,
    used: Vec<bool>,
    size: usize,
    rand: rand::prelude::ThreadRng,
}

impl RandPoll {
    pub fn new(size: usize) -> Self {
        let mut s = Self {
            v: Vec::with_capacity(size),
            used: Vec::with_capacity(size),
            size: size,
            rand: rand::thread_rng(),
        };

        for i in 0..size {
            s.v.push(i);
            s.used.push(false);
        }
        return s;
    }

    pub fn is_empty(&self) -> bool {
        return self.size == 0;
    }

    pub fn size(&self) -> usize {
        return self.size;
    }

    pub fn next(&mut self) -> usize {
        let res = self.next0();
        if self.size <= self.v.len() / 2 {
            self.v.retain(|x| !self.used[*x]);
        }
        return res;
    }

    fn next0(&mut self) -> usize {
        assert!(!self.is_empty());
        loop {
            let x = self.v[self.rand.gen_range(0..self.v.len())];
            if !self.used[x] {
                self.used[x] = true;
                self.size -= 1;
                return x;
            }
        }
    }
}
