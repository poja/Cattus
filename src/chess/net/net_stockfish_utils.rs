use pleco::board::castle_rights::Castling;
use pleco::core::masks::*;
use pleco::core::mono_traits::*;
use pleco::core::score::*;
use pleco::core::*;
use pleco::helper::prelude::*;
use pleco::BitBoard;
use pleco::File;
use pleco::Rank;
use pleco::SQ;
use pleco::{Board, PieceType, Player};
use std::mem::transmute;

// pub const PHASE_END_GAME: u16 = 0;
pub const PHASE_MID_GAME: u16 = 128;

pub const SCALE_FACTOR_DRAW: u8 = 0;
pub const SCALE_FACTOR_ONEPAWN: u8 = 48;
pub const SCALE_FACTOR_NORMAL: u8 = 64;
// pub const SCALE_FACTOR_MAX: u8 = 128;
// pub const SCALE_FACTOR_NONE: u8 = 255;

// Polynomial material imbalance parameters
const QUADRATIC_OURS: [[i32; PIECE_TYPE_CNT - 2]; PIECE_TYPE_CNT - 2] = [
    [1667, 0, 0, 0, 0, 0],           // Bishop pair
    [40, 0, 0, 0, 0, 0],             // Pawn
    [32, 255, -3, 0, 0, 0],          // Knight      OUR PIECES
    [0, 104, 4, 0, 0, 0],            // Bishop
    [-26, -2, 47, 105, -149, 0],     // Rook
    [-189, 24, 117, 133, -134, -10], // Queen
]; // pair pawn knight bishop rook queen
   //            OUR PIECES

const QUADRATIC_THEIRS: [[i32; PIECE_TYPE_CNT - 2]; PIECE_TYPE_CNT - 2] = [
    [0, 0, 0, 0, 0, 0],          // Bishop pair
    [36, 0, 0, 0, 0, 0],         // Pawn
    [9, 63, 0, 0, 0, 0],         // Knight      OUR PIECES
    [59, 65, 42, 0, 0, 0],       // Bishop
    [46, 39, 24, -24, 0, 0],     // Rook
    [97, 100, -42, 137, 268, 0], // Queen
]; // pair pawn knight bishop rook queen
   //           THEIR PIECES

pub struct MaterialEntry {
    pub value: Value,
    pub factor: [u8; PLAYER_CNT],
    pub phase: u16,
}

impl MaterialEntry {
    pub fn new(board: &Board) -> Self {
        let mut factor = [SCALE_FACTOR_NORMAL; PLAYER_CNT];

        let npm_w: Value = board.non_pawn_material(Player::White);
        let npm_b: Value = board.non_pawn_material(Player::Black);
        let npm: Value = END_GAME_LIMIT.max(MID_GAME_LIMIT.min(npm_w + npm_b));

        let phase = (((npm - END_GAME_LIMIT) * PHASE_MID_GAME as i32)
            / (MID_GAME_LIMIT - END_GAME_LIMIT)) as u16;

        let w_pawn_count: u8 = board.count_piece(Player::White, PieceType::P);
        let w_knight_count: u8 = board.count_piece(Player::White, PieceType::N);
        let w_bishop_count: u8 = board.count_piece(Player::White, PieceType::B);
        let w_rook_count: u8 = board.count_piece(Player::White, PieceType::R);
        let w_queen_count: u8 = board.count_piece(Player::White, PieceType::Q);

        let b_pawn_count: u8 = board.count_piece(Player::Black, PieceType::P);
        let b_knight_count: u8 = board.count_piece(Player::Black, PieceType::N);
        let b_bishop_count: u8 = board.count_piece(Player::Black, PieceType::B);
        let b_rook_count: u8 = board.count_piece(Player::Black, PieceType::R);
        let b_queen_count: u8 = board.count_piece(Player::Black, PieceType::Q);

        if w_pawn_count == 0 && npm_w - npm_b <= BISHOP_MG {
            factor[Player::White as usize] = if npm_w < ROOK_MG {
                SCALE_FACTOR_DRAW
            } else if npm_b <= BISHOP_MG {
                4
            } else {
                14
            };
        }

        if b_pawn_count == 0 && npm_b - npm_w <= BISHOP_MG {
            factor[Player::Black as usize] = if npm_b < ROOK_MG {
                SCALE_FACTOR_DRAW
            } else if npm_w <= BISHOP_MG {
                4
            } else {
                14
            };
        }

        if w_pawn_count == 1 && npm_w - npm_b <= BISHOP_MG {
            factor[Player::White as usize] = SCALE_FACTOR_ONEPAWN;
        }

        if b_pawn_count == 1 && npm_b - npm_w <= BISHOP_MG {
            factor[Player::Black as usize] = SCALE_FACTOR_ONEPAWN;
        }

        let w_pair_bish: u8 = (w_bishop_count > 1) as u8;
        let b_pair_bish: u8 = (b_bishop_count > 1) as u8;

        let piece_counts: [[u8; PIECE_TYPE_CNT - 2]; PLAYER_CNT] = [
            [
                w_pair_bish,
                w_pawn_count,
                w_knight_count,
                w_bishop_count,
                w_rook_count,
                w_queen_count,
            ],
            [
                b_pair_bish,
                b_pawn_count,
                b_knight_count,
                b_bishop_count,
                b_rook_count,
                b_queen_count,
            ],
        ];

        let value =
            (imbalance::<WhiteType>(&piece_counts) - imbalance::<BlackType>(&piece_counts)) / 16;

        return Self {
            value: value,
            factor: factor,
            phase: phase,
        };
    }

    #[inline(always)]
    pub fn score(&self) -> Score {
        Score(self.value, self.value)
    }

    #[inline(always)]
    pub fn scale_factor(&self, player: Player) -> u8 {
        self.factor[player as usize]
    }
}

fn imbalance<P: PlayerTrait>(piece_counts: &[[u8; PIECE_TYPE_CNT - 2]; PLAYER_CNT]) -> i32 {
    let mut bonus: i32 = 0;

    for pt1 in 0..6 {
        if piece_counts[P::player() as usize][pt1] == 0 {
            continue;
        }

        let mut v: i32 = 0;

        for pt2 in 0..6 {
            v += QUADRATIC_OURS[pt1][pt2] * piece_counts[P::player() as usize][pt2] as i32
                + QUADRATIC_THEIRS[pt1][pt2] * piece_counts[P::opp_player() as usize][pt2] as i32;
        }

        bonus += piece_counts[P::player() as usize][pt1] as i32 * v;
    }
    bonus
}

const ISOLATED: Score = Score(13, 18);

// backwards pawn penalty
const BACKWARDS: Score = Score(24, 12);

// doubled pawn penalty
const DOUBLED: Score = Score(18, 28);

// Lever bonus by rank
// const LEVER: [Score; RANK_CNT] = [
//     Score(0, 0),
//     Score(0, 0),
//     Score(0, 0),
//     Score(0, 0),
//     Score(17, 16),
//     Score(33, 32),
//     Score(0, 0),
//     Score(0, 0),
// ];

const MAX_SAFETY_BONUS: Value = 258;

// Weakness of our pawn shelter in front of the king by [isKingFile][distance from edge][rank].
// RANK_1 = 0 is used for files where we have no pawns or our pawn is behind our king.
const SHELTER_WEAKNESS: [[[Value; RANK_CNT]; 4]; 2] = [
    [
        [0, 97, 17, 9, 44, 84, 87, 99], // Not On King file
        [0, 106, 6, 33, 86, 87, 104, 112],
        [0, 101, 2, 65, 98, 58, 89, 115],
        [0, 73, 7, 54, 73, 84, 83, 111],
    ],
    [
        [0, 104, 20, 6, 27, 86, 93, 82], // On King file
        [0, 123, 9, 34, 96, 112, 88, 75],
        [0, 120, 25, 65, 91, 66, 78, 117],
        [0, 81, 2, 47, 63, 94, 93, 104],
    ],
];

// Danger of enemy pawns moving toward our king by [type][distance from edge][rank].
// For the unopposed and unblocked cases, RANK_1 = 0 is used when opponent has
// no pawn on the given file, or their pawn is behind our king.
const STORM_DANGER: [[[Value; RANK_CNT]; 4]; 4] = [
    [
        [0, -290, -274, 57, 41, 0, 0, 0], // BlockedByKing
        [0, 60, 144, 39, 13, 0, 0, 0],
        [0, 65, 141, 41, 34, 0, 0, 0],
        [0, 53, 127, 56, 14, 0, 0, 0],
    ],
    [
        [4, 73, 132, 46, 31, 0, 0, 0], // Unopposed
        [1, 64, 143, 26, 13, 0, 0, 0],
        [1, 47, 110, 44, 24, 0, 0, 0],
        [0, 72, 127, 50, 31, 0, 0, 0],
    ],
    [
        [0, 0, 79, 23, 1, 0, 0, 0], // BlockedByPawn
        [0, 0, 148, 27, 2, 0, 0, 0],
        [0, 0, 161, 16, 1, 0, 0, 0],
        [0, 0, 171, 22, 15, 0, 0, 0],
    ],
    [
        [22, 45, 104, 62, 6, 0, 0, 0], // Unblocked
        [31, 30, 99, 39, 19, 0, 0, 0],
        [23, 29, 96, 41, 15, 0, 0, 0],
        [21, 23, 116, 41, 15, 0, 0, 0],
    ],
];

pub static mut CONNECTED: [[[[Score; RANK_CNT]; 3]; 2]; 2] = [[[[Score(0, 0); RANK_CNT]; 3]; 2]; 2];

/// Initalizes the CONNECTED table.
// #[cold]
// pub fn init() {
//     unsafe {
//         let seed: [i32; 8] = [0, 13, 24, 18, 76, 100, 175, 330];
//         for opposed in 0..2 {
//             for phalanx in 0..2 {
//                 for support in 0..3 {
//                     for r in 1..7 {
//                         let mut v: i32 = 17 * support;
//                         v += (seed[r]
//                             + (phalanx * ((seed[r as usize + 1] - seed[r as usize]) / 2)))
//                             >> opposed;
//                         let eg: i32 = v * (r as i32 - 2) / 4;
//                         CONNECTED[opposed as usize][phalanx as usize][support as usize]
//                             [r as usize] = Score(v, eg);
//                     }
//                 }
//             }
//         }
//     }
// }

// fn init_connected() -> [[[[Score; 2]; 2]; 3]; RANK_CNT] {
//     let seed: [i32; 8] = [0, 13, 24, 18, 76, 100, 175, 330];
//     let mut a: [[[[Score; 2]; 2]; 3]; 8] = [[[[Score(0, 0); 2]; 2]; 3]; 8];
//     for opposed in 0..2 {
//         for phalanx in 0..2 {
//             for support in 0..3 {
//                 for r in 1..7 {
//                     let mut v: i32 = 17 * support;
//                     v += (seed[r] + (phalanx * ((seed[r as usize + 1] - seed[r as usize]) / 2)))
//                         >> opposed;
//                     let eg: i32 = v * (r as i32 - 2) / 4;
//                     a[r as usize][support as usize][phalanx as usize][opposed as usize] =
//                         Score(v, eg);
//                 }
//             }
//         }
//     }
//     a
// }

/// Information on a the pawn structure for a given position.
///
/// This information is computed upon access.
pub struct PawnEntry {
    score: [Score; PLAYER_CNT],
    passed_pawns: [BitBoard; PLAYER_CNT],
    pawn_attacks: [BitBoard; PLAYER_CNT],
    pawn_attacks_span: [BitBoard; PLAYER_CNT],
    king_squares: [SQ; PLAYER_CNT],
    king_safety_score: [Score; PLAYER_CNT],
    weak_unopposed: [i16; PLAYER_CNT],
    castling_rights: [Castling; PLAYER_CNT],
    semiopen_files: [u8; PLAYER_CNT],
    // per
    pawns_on_squares: [[u8; PLAYER_CNT]; PLAYER_CNT], // [color][light/dark squares]
    asymmetry: i16,
    open_files: u8,
}

impl PawnEntry {
    pub fn new(board: &Board) -> Self {
        let mut entry = Self {
            score: [Score::ZERO; PLAYER_CNT],
            passed_pawns: [BitBoard::ALL; PLAYER_CNT],
            pawn_attacks: [BitBoard::ALL; PLAYER_CNT],
            pawn_attacks_span: [BitBoard::ALL; PLAYER_CNT],
            king_squares: [SQ::NONE; PLAYER_CNT],
            king_safety_score: [Score::ZERO; PLAYER_CNT],
            weak_unopposed: [0; PLAYER_CNT],
            castling_rights: [Castling::WHITE_ALL | Castling::BLACK_ALL; PLAYER_CNT],
            semiopen_files: [0; PLAYER_CNT],
            // per
            pawns_on_squares: [[0; PLAYER_CNT]; PLAYER_CNT], // [color][light/dark squares]
            asymmetry: 0,
            open_files: 0,
        };

        entry.score[Player::White as usize] = entry.evaluate::<WhiteType>(board);
        entry.score[Player::Black as usize] = entry.evaluate::<BlackType>(board);
        entry.open_files = (entry.semiopen_files[Player::White as usize]
            & entry.semiopen_files[Player::Black as usize])
            .count_ones() as u8;

        let all_passed: BitBoard =
            entry.passed_pawns[Player::White as usize] | entry.passed_pawns[Player::Black as usize];
        let exclusive_open_files = entry.semiopen_files[Player::White as usize]
            ^ entry.semiopen_files[Player::Black as usize];

        entry.asymmetry = (all_passed | BitBoard(exclusive_open_files as u64)).count_bits() as i16;

        return entry;
    }

    /// Returns the current score of the pawn structure.
    #[inline(always)]
    pub fn pawns_score(&self, player: Player) -> Score {
        self.score[player as usize]
    }

    /// Returns the possible pawn attacks `BitBoard` of a player.
    #[inline(always)]
    pub fn pawn_attacks(&self, player: Player) -> BitBoard {
        self.pawn_attacks[player as usize]
    }

    /// Returns the `BitBoard` of the passed pawns for a specified player. A passed pawn is one that
    /// has no opposing pawns in the same file, or any adjacent file.
    #[inline(always)]
    pub fn passed_pawns(&self, player: Player) -> BitBoard {
        self.passed_pawns[player as usize]
    }

    /// Returns the span of all the pawn's attacks for a given player.
    #[inline(always)]
    pub fn pawn_attacks_span(&self, player: Player) -> BitBoard {
        self.pawn_attacks_span[player as usize]
    }

    /// Returns the weak-unopposed score of the given player. This measures the strength of the pawn
    /// structure when considering isolated and disconnected pawns.
    #[inline(always)]
    pub fn weak_unopposed(&self, player: Player) -> i16 {
        self.weak_unopposed[player as usize]
    }

    /// Assymetric score of a position.
    #[inline(always)]
    pub fn asymmetry(&self) -> i16 {
        self.asymmetry
    }

    /// Returns a bitfield of the current ranks.
    #[inline(always)]
    pub fn open_files(&self) -> u8 {
        self.open_files
    }

    /// Returns if a file is semi-open for a given player, meaning there are no pieces of the
    /// opposing player on that file.
    #[inline]
    pub fn semiopen_file(&self, player: Player, file: File) -> bool {
        self.semiopen_files[player as usize] & (1 << file as u8) != 0
    }

    /// Returns if a side of a file is semi-open, meaning no enemy pieces.
    // #[inline]
    // pub fn semiopen_side(&self, player: Player, file: File, left_side: bool) -> bool {
    //     let side_mask: u8 = if left_side {
    //         file.left_side_mask()
    //     } else {
    //         file.right_side_mask()
    //     };
    //     self.semiopen_files[player as usize] & side_mask != 0
    // }

    // returns count of pawns of a player on the same color square as the player's color.
    #[inline]
    pub fn pawns_on_same_color_squares(&self, player: Player, sq: SQ) -> u8 {
        self.pawns_on_squares[player as usize][sq.square_color_index()]
    }

    /// Returns the current king safety `Score` for a given player and king square.
    pub fn king_safety<P: PlayerTrait>(&mut self, board: &Board, ksq: SQ) -> Score {
        if self.king_squares[P::player_idx()] == ksq
            && self.castling_rights[P::player_idx()] == board.player_can_castle(P::player())
        {
            self.king_safety_score[P::player_idx()]
        } else {
            self.king_safety_score[P::player_idx()] = self.do_king_safety::<P>(board, ksq);
            self.king_safety_score[P::player_idx()]
        }
    }

    fn do_king_safety<P: PlayerTrait>(&mut self, board: &Board, ksq: SQ) -> Score {
        self.king_squares[P::player_idx()] = ksq;
        self.castling_rights[P::player_idx()] = board.player_can_castle(P::player());
        let mut min_king_distance = 0;

        let pawns: BitBoard = board.piece_bb(P::player(), PieceType::P);
        if !pawns.is_empty() {
            while (ring_distance(ksq, min_king_distance as u8) & pawns).is_empty() {
                min_king_distance += 1;
            }
        }

        let mut bonus: Value = self.shelter_storm::<P>(board, ksq);

        if board.can_castle(P::player(), CastleType::KingSide) {
            bonus = bonus.max(self.shelter_storm::<P>(board, P::player().relative_square(SQ::G1)));
        }

        if board.can_castle(P::player(), CastleType::QueenSide) {
            bonus = bonus.max(self.shelter_storm::<P>(board, P::player().relative_square(SQ::C1)));
        }

        Score::new(bonus, -16 * min_king_distance)
    }

    fn shelter_storm<P: PlayerTrait>(&self, board: &Board, ksq: SQ) -> Value {
        let center: File = (File::B).max(File::G.min(ksq.file()));

        let mut b: BitBoard = board.piece_bb_both_players(PieceType::P)
            & (forward_rank_bb(P::player(), ksq.rank()) | ksq.rank_bb())
            & (adjacent_file(center) | SQ(center as u8).file_bb());

        let our_pawns: BitBoard = b & board.get_occupied_player(P::player());
        let their_pawns: BitBoard = b & board.get_occupied_player(P::opp_player());
        let mut safety: Value = MAX_SAFETY_BONUS;

        for file in ((center as u8) - 1)..((center as u8) + 2) {
            b = our_pawns & SQ(file).file_bb();
            let rk_us: Rank = if b.is_empty() {
                Rank::R1
            } else {
                P::player().relative_rank_of_sq(b.backmost_sq(P::player()))
            };

            b = their_pawns & SQ(file).file_bb();
            let rk_them: Rank = if b.is_empty() {
                Rank::R1
            } else {
                P::player().relative_rank_of_sq(b.frontmost_sq(P::opp_player()))
            };
            let d: File =
                unsafe { (transmute::<u8, File>(file)).min(!transmute::<u8, File>(file)) };

            // TODO: Simplify
            let r = if file == ksq.file() as u8 { 1 } else { 0 };

            let storm_danger_idx: usize = if file == ksq.file() as u8
                && P::player().relative_rank_of_sq(ksq) as u8 + 1 == rk_them as u8
            {
                0 // Blocked By King
            } else if rk_us == Rank::R1 {
                1 // Unopossed
            } else if rk_them as u8 == rk_us as u8 + 1 {
                2 // Blocked by Pawn
            } else {
                3 // Unblocked
            };

            safety -= SHELTER_WEAKNESS[r as usize][d as usize][rk_us as usize];
            safety -= STORM_DANGER[storm_danger_idx][d as usize][rk_them as usize];
        }
        safety
    }

    fn evaluate<P: PlayerTrait>(&mut self, board: &Board) -> Score {
        let mut b: BitBoard;
        let mut neighbours: BitBoard;
        let mut stoppers: BitBoard;
        let mut doubled: BitBoard;
        let mut supported: BitBoard;
        let mut phalanx: BitBoard;
        let mut lever: BitBoard;
        let mut lever_push: BitBoard;
        let mut opposed: bool;
        let mut backward: bool;

        let mut score: Score = Score::ZERO;
        let our_pawns: BitBoard = board.piece_bb(P::player(), PieceType::P);
        let their_pawns: BitBoard = board.piece_bb(P::opp_player(), PieceType::P);

        let mut p1: BitBoard = our_pawns;

        self.passed_pawns[P::player() as usize] = BitBoard(0);
        self.pawn_attacks_span[P::player() as usize] = BitBoard(0);
        self.weak_unopposed[P::player() as usize] = 0;
        self.semiopen_files[P::player() as usize] = 0xFF;
        self.king_squares[P::player() as usize] = SQ::NO_SQ;
        self.pawn_attacks[P::player() as usize] =
            P::shift_up_left(our_pawns) | P::shift_up_right(our_pawns);

        let pawns_on_dark: u8 = (our_pawns & BitBoard::DARK_SQUARES).count_bits();
        self.pawns_on_squares[P::player() as usize][Player::Black as usize] = pawns_on_dark;
        self.pawns_on_squares[P::player() as usize][Player::White as usize] =
            board.count_piece(P::player(), PieceType::P) - pawns_on_dark;

        while let Some(s) = p1.pop_some_lsb() {
            assert_eq!(
                board.piece_at_sq(s),
                Piece::make_lossy(P::player(), PieceType::P)
            );

            let f: File = s.file();

            self.semiopen_files[P::player() as usize] &= !(1 << f as u8);
            self.pawn_attacks[P::player() as usize] |= pawn_attacks_span(P::player(), s);

            opposed = (their_pawns & forward_file_bb(P::player(), s)).is_not_empty();
            stoppers = their_pawns & passed_pawn_mask(P::player(), s);
            lever = their_pawns & pawn_attacks_from(s, P::player());
            lever_push = their_pawns & pawn_attacks_from(P::up(s), P::player());
            doubled = our_pawns & (P::down(s)).to_bb();
            neighbours = our_pawns & adjacent_file(f);
            phalanx = neighbours & s.rank_bb();
            supported = neighbours & (P::down(s)).rank_bb();

            // A pawn is backward when it is behind all pawns of the same color on the
            // adjacent files and cannot be safely advanced.
            if neighbours.is_empty()
                || lever.is_not_empty()
                || P::player().relative_rank_of_sq(s) >= Rank::R5
            {
                backward = false;
            } else {
                // Find the backmost rank with neighbours or stoppers
                b = (neighbours | stoppers).backmost_sq(P::player()).rank_bb();

                // The pawn is backward when it cannot safely progress to that rank:
                // either there is a stopper in the way on this rank, or there is a
                // stopper on adjacent file which controls the way to that rank.
                backward = ((b | P::shift_up(b & adjacent_file(f))) & stoppers).is_not_empty();

                assert!(
                    !(backward
                        && (forward_rank_bb(P::opp_player(), P::up(s).rank()) & neighbours)
                            .is_not_empty())
                );
            }

            // Passed pawns will be properly scored in evaluation because we need
            // full attack info to evaluate them. Include also not passed pawns
            // which could become passed after one or two pawn pushes when are
            // not attacked more times than defended.
            if (stoppers ^ lever ^ lever_push).is_empty()
                && (our_pawns & forward_file_bb(P::player(), s)).is_empty()
                && supported.count_bits() as i8 >= lever.count_bits() as i8 - 1
                && phalanx.count_bits() >= lever_push.count_bits()
            {
                self.passed_pawns[P::player() as usize] |= s.to_bb();
            } else if stoppers == P::up(s).to_bb() && P::player().relative_rank_of_sq(s) >= Rank::R5
            {
                b = P::shift_up(supported) & !their_pawns;
                while let Some(b_sq) = b.pop_some_lsb() {
                    if !(their_pawns & pawn_attacks_from(b_sq, P::player())).more_than_one() {
                        self.passed_pawns[P::player() as usize] |= s.to_bb();
                    }
                }
            }

            if supported.is_not_empty() | supported.is_not_empty() {
                score += unsafe {
                    CONNECTED[opposed as usize][phalanx.is_not_empty() as usize]
                        [supported.count_bits() as usize]
                        [P::player().relative_rank_of_sq(s) as usize]
                };
            } else if neighbours.is_empty() {
                score -= ISOLATED;
                self.weak_unopposed[P::player() as usize] += (!opposed) as i16;
            } else if backward {
                score -= BACKWARDS;
                self.weak_unopposed[P::player() as usize] += (!opposed) as i16;
            }

            if doubled.is_not_empty() && supported.is_empty() {
                score -= DOUBLED;
            }
        }
        score
    }
}
