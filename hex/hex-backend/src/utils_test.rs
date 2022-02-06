#[cfg(test)]
mod tests {
    use crate::utils;
    use std::collections;

    #[test]
    fn rand_poll_test() {
        for (repeat, size) in [(128, 4), (64, 16), (16, 37), (4, 1024)] {
            for _ in 0..repeat {
                let mut poll = utils::RandPoll::new(size);
                let mut seen = collections::HashSet::with_capacity(size);
                let mut unseen = collections::HashSet::with_capacity(size);
                for x in 0..size {
                    unseen.insert(x);
                }

                while !poll.is_empty() {
                    let x = poll.next();
                    assert!(!seen.contains(&x));
                    seen.insert(x);
                    assert!(unseen.contains(&x));
                    unseen.remove(&x);
                }
                assert!(unseen.is_empty());
            }
        }
    }
}
