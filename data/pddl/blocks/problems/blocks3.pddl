;; 866 states
;; 5 blocks
(define (problem medium)
(:domain BLOCKS)
(:objects A B C D E - block)
(:INIT (ONTABLE A) (ONTABLE C) (ON B C) (ON D A) (ON E D) (CLEAR E) (CLEAR B) (HANDEMPTY))
(:goal (AND (ON D C) (ON C A) (ON A B)))
)