/** usecase_table_metrics_v2.sc (fixed)
 * Static use-case oriented metrics over Joern CPG for C/C++.
 * Keeps v1 metrics + your research metrics (BRCG-ish, CFG stats, domain terms, TF-IDF, etc.).
 *
 * Env:
 *   CPG_PATH  (default: /work/out/cpg.bin)
 *   OUT_JSON  (default: /work/out/usecase_table_metrics_v2.json)
 *   UCD_TERMS (optional, comma-separated list of UCD terms to match)
 */

import io.shiftleft.codepropertygraph.cpgloading.CpgLoader
import io.shiftleft.codepropertygraph.generated.nodes._
import io.shiftleft.semanticcpg.language._
import io.shiftleft.semanticcpg.language.locationCreator   // line/locations helpers
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
import scala.collection.mutable
import scala.util.Try

// ---------- Config ----------
val cpgPath = sys.env.getOrElse("CPG_PATH", "/work/out/cpg.bin")
val outPath = sys.env.getOrElse("OUT_JSON", "/work/out/usecase_table_metrics_v2.json")
val highCouplingThreshold = 10
val ucdTerms: Seq[String] = sys.env.get("UCD_TERMS").map(_.split(",").toSeq.map(_.trim).filter(_.nonEmpty)).getOrElse(Seq.empty)

// ---------- Load ----------
val cpg = CpgLoader.load(cpgPath)

// ---------- JSON helpers ----------
def esc(s: String): String =
  Option(s).getOrElse("").flatMap {
    case '"'  => "\\\""
    case '\\' => "\\\\"
    case '\b' => "\\b"
    case '\f' => "\\f"
    case '\n' => "\\n"
    case '\r' => "\\r"
    case '\t' => "\\t"
    case c if c < ' ' => "\\u%04x".format(c.toInt)
    case c => s"$c"
  }
def jstr(s:String) = "\"" + esc(s) + "\""
def jbool(b:Boolean) = if (b) "true" else "false"
def jint(i:Int) = i.toString
def jarr[T](xs:Iterable[T])(f:T=>String) = xs.map(f).mkString("[",",","]")
def jmap[V](m:Map[String,V])(f:V=>String) = "{"+m.toSeq.sortBy(_._1).map{case(k,v)=> jstr(k)+":"+f(v)}.mkString(",")+"}"
def baseName(p:String) = p.split("[/\\\\]").lastOption.getOrElse(p)

// ---------- Recognizers ----------
val entryPointNames = List("main","WinMain","DllMain")
val ioFile    = Set("fopen","freopen","fclose","fread","fwrite","fprintf","fscanf","open","close","read","write")
val ioConsole = Set("printf","puts","scanf","gets","getchar","putchar","cin","cout","cerr")
val ioNet     = Set("socket","connect","send","recv","bind","listen","accept","sendto","recvfrom")
val ioEnv     = Set("getenv","putenv","setenv","unsetenv")
val knownUnsafe = Set("gets","strcpy","strcat","sprintf","vsprintf","scanf")
val exitFuncs   = Set("exit","abort","_exit","quick_exit")
val nameClues   = List("login","logout","auth","register","create","delete","update","add","remove","deposit","withdraw","transfer","search","upload","download")
val inputNames  = Set("argv","argc","input","buf","line","user","password","username")

// ---------- Functions (robust) ----------
val allMethods = cpg.method.l
val definedMethods: List[Method] = allMethods.filter(m => m.ast.isBlock.l.nonEmpty || !m.isExternal)

val functionNames     = definedMethods.map(_.name).filter(_.nonEmpty).distinct.sorted
val functionFullNames = definedMethods.map(_.fullName).filter(_.nonEmpty).distinct.sorted
val functionSigs      = definedMethods.flatMap(m => Option(m.signature).map(_.toString)).filter(_.nonEmpty).distinct.sorted

// ---------- Entry points ----------
val entryMethods   = cpg.method.nameExact(entryPointNames:_*).l
val publicMethods  = cpg.method.where(_.modifier.modifierType("PUBLIC")).l
val entryPointFullNames = (entryMethods.map(_.fullName) ++ publicMethods.map(_.fullName)).distinct.sorted

// ---------- Call graph (caller → callee) ----------
case class Edge(caller:String, callee:String, file:String, line:Option[Int])
val edges: List[Edge] = cpg.call.l.map { call =>
  val caller = call.method.fullName
  val callee = Option(call.methodFullName).getOrElse(call.name)
  val file   = call.file.name.headOption.getOrElse("")
  val line   = call.lineNumber.map(_.toInt)
  Edge(caller, callee, file, line)
}.distinct

val callersByCallee: Map[String,Set[String]] = edges.groupBy(_.callee).view.mapValues(_.map(_.caller).toSet).toMap
val calleesByCaller: Map[String,Set[String]] = edges.groupBy(_.caller).view.mapValues(_.map(_.callee).toSet).toMap

// ---------- BRCG-ish: Branch statements & call connectivity ----------
def isIfCode(s:String) = s.trim.toLowerCase.startsWith("if")
def isSwitchCode(s:String) = s.trim.toLowerCase.startsWith("switch")
def isLoopCode(s:String) = { val t=s.trim.toLowerCase; t.startsWith("for")||t.startsWith("while")||t.startsWith("do") }

case class BranchInfo(kind:String, code:String, file:String, line:Option[Int])
def branchInfos(m:Method): List[BranchInfo] = {
  val cs = m.controlStructure.l
  cs.map { n =>
    val c = Option(n.code).getOrElse("")
    val kind =
      if (isIfCode(c)) "if"
      else if (isSwitchCode(c)) "switch"
      else if (isLoopCode(c)) "loop"
      else if (c.trim.toLowerCase.startsWith("else")) "else"
      else "other"
    BranchInfo(kind, c, n.file.name.headOption.getOrElse(""), n.lineNumber.map(_.toInt))
  }
}

// Heuristic pruning of branches: keep "meaningful" ones
val branchKeywords = Set("error","fail","invalid","retry","menu","option","case","success","ok","auth","login","permission","granted","denied")
def meaningfulBranch(b:BranchInfo, m:Method): Boolean = {
  val low = b.code.toLowerCase
  val mentionsInput = inputNames.exists(low.contains)
  val mentionsKeyword = branchKeywords.exists(low.contains)
  val methodHasIO = m.call.name.l.exists(n => ioConsole(n.toLowerCase) || ioFile(n.toLowerCase) || ioNet(n.toLowerCase) || ioEnv(n.toLowerCase))
  mentionsInput || mentionsKeyword || methodHasIO
}

// ---------- CFG traversal approximations ----------
case class CfgStats(nodes:Int, edges:Int, backEdges:Int, longestAcyclic:Int, basicSegments:Int)

def cfgStats(m:Method): CfgStats = {
  // فقط با id کار می‌کنیم، نه با خودِ CfgNode
  val nodes: List[CfgNode] = m.cfgNode.l.collect { case n: CfgNode => n }.distinct
  val ids: Set[Long] = nodes.map(_.id).toSet

  // یال‌های CFG به‌صورت (fromId,toId)
  val rawEdges: List[(Long, Long)] =
    nodes.flatMap(n => n.cfgNext.l.collect{ case nn: CfgNode => (n.id, nn.id) })

  // back-edge heuristic: successor.line < current.line
  def ln(id: Long): Option[Int] =
    nodes.find(_.id == id).flatMap(_.lineNumber).map(_.toInt)

  val backEdges: Int = rawEdges.count { case (a,b) =>
    (for { la <- ln(a); lb <- ln(b) } yield lb < la).getOrElse(false)
  }

  // DAG تقریبی: حذف back-edge ها و ساخت adjacency با id
  val dag: Map[Long, Set[Long]] =
    rawEdges
      .filterNot { case (a,b) => (for { la <- ln(a); lb <- ln(b) } yield lb < la).getOrElse(false) }
      .groupBy(_._1).view.mapValues(_.map(_._2).toSet).toMap

  // DFS با memo + seen برای جلوگیری از حلقه
  val memo = scala.collection.mutable.HashMap.empty[Long, Int]
  def dfs(u: Long, seen: Set[Long]): Int = {
    if (seen.contains(u)) 0
    else memo.getOrElseUpdate(
      u, dag.getOrElse(u, Set.empty).map(v => 1 + dfs(v, seen + u)).foldLeft(0)(Math.max)
    )
  }

  val longest: Int = ids.map(id => dfs(id, Set.empty)).foldLeft(0)(Math.max)

  // basic segments: همان منطق قبلی ولی بر مبنای درجات ورود/خروجِ idها
  val outDeg: Map[Long, Int] = rawEdges.groupBy(_._1).view.mapValues(_.size).toMap
  val inDeg : Map[Long, Int] = rawEdges.groupBy(_._2).view.mapValues(_.size).toMap

  def isBranchNodeId(id: Long): Boolean = {
    val code = nodes.find(_.id == id).flatMap(n => Option(n.code)).getOrElse("").trim.toLowerCase
    code.startsWith("if") || code.startsWith("switch") || code.startsWith("case") || code.startsWith("default")
  }

  val basicSegments = ids.count { id =>
    val o = outDeg.getOrElse(id, 0); val i = inDeg.getOrElse(id, 0)
    (i != 1) || isBranchNodeId(id)
  }

  CfgStats(nodes.size, rawEdges.size, backEdges, longest, basicSegments)
}


// ---------- Control Flow (project totals) ----------
val branchCount = cpg.controlStructure.code.l.count(isIfCode)
val loopCount   = cpg.controlStructure.code.l.count(isLoopCode)
val switchCount = cpg.controlStructure.code.l.count(isSwitchCode)
val errorExitTotal = cpg.call.nameExact(exitFuncs.toSeq:_*).size

// ---------- Data / Artifacts ----------
val cliArgsUse = cpg.identifier.nameExact("argv","argc").size

def locals(m:Method) = m.local.name.l.toSet
def globalsUsed(m:Method) = {
  val locs = locals(m); val pars = m.parameter.name.l.toSet
  m.ast.isIdentifier.name.l.filterNot(n => locs.contains(n) || pars.contains(n) || n=="this").toSet
}
val sharedVarToFuncs: Map[String,Set[String]] =
  definedMethods.foldLeft(Map.empty[String, Set[String]]){ (acc,m) =>
    globalsUsed(m).foldLeft(acc){ case (a, g) => a.updated(g, a.getOrElse(g, Set.empty) + m.fullName) }
  }
val sharedArtifacts = sharedVarToFuncs.keySet.toSeq.sorted

def stringArgsOf(names:Set[String]) =
  cpg.call.nameExact(names.toSeq:_*).argument.isLiteral.code.l
    .map(_.stripPrefix("\"").stripSuffix("\""))
val filesTouched     = stringArgsOf(ioFile).distinct.sorted
val networkEndpoints = stringArgsOf(ioNet).distinct.sorted

// ---------- I/O counts ----------
val consoleInCount  = cpg.call.nameExact("scanf","gets","getchar","cin").size
val consoleOutCount = cpg.call.nameExact("printf","puts","putchar","cout","cerr").size
val fileIOCount     = cpg.call.name.l.count(ioFile.contains)
val netIOCount      = cpg.call.name.l.count(ioNet.contains)
val envIOCount      = cpg.call.name.l.count(ioEnv.contains)

// ---------- Name/Text Cues + Domain terms ----------
def splitTokens(s:String): Seq[String] = {
  val basic = s.replaceAll("[^A-Za-z0-9_]+"," ")
    .replaceAll("([a-z])([A-Z])","$1 $2")
    .toLowerCase.split("\\s+").toSeq.filter(_.nonEmpty)
  basic.flatMap(_.split("_")).filter(t => t.length>=3)
}
val stop = Set("int","char","void","long","short","double","float","const","static","class","struct","return","include","using","std","namespace","bool","true","false","null")

// comments inside method span — get comments by file via File → AST → Comment
def functionComments(m: Method): Seq[String] = {
  val f     = m.file.name.headOption.getOrElse("")
  val start = m.lineNumber.map(_.toInt).getOrElse(Int.MinValue)
  val end   = m.lineNumberEnd.map(_.toInt).getOrElse(Int.MaxValue)

  cpg.comment.l
    .filter { c =>
      // filename/lineNumber provided by `import io.shiftleft.semanticcpg.language.locationCreator`
      val sameFile =
        scala.util.Try(c.filename).toOption.contains(f) ||         // preferred
        scala.util.Try(c.location.filename).toOption.contains(f)   // fallback on some builds
      val inSpan   = c.lineNumber.exists(ln => ln >= start && ln <= end)
      sameFile && inSpan
    }
    .flatMap(c => Option(c.code))
}


// field identifier names (prefer canonicalName)
def memberAccessNames(m:Method): Seq[String] =
  m.ast.isFieldIdentifier.l.flatMap(fi => Option(fi.canonicalName))

def bagOfWords(m:Method): Seq[String] = {
  val idents = m.ast.isIdentifier.name.l
  val names  = Seq(m.name) ++ memberAccessNames(m)
  val cmts   = functionComments(m)
  (idents ++ names ++ cmts).flatMap(splitTokens).filterNot(stop)
}

// TF-IDF per function
val funcBags: Map[String, Seq[String]] = definedMethods.map(m => m.fullName -> bagOfWords(m)).toMap
val df: Map[String, Int] = funcBags.values.flatMap(_.distinct).toSeq.groupBy(identity).view.mapValues(_.size).toMap
val Ndocs = funcBags.size.max(1)
def tfidf(fn:String, k:String): Double = {
  val tf = funcBags.getOrElse(fn, Seq.empty).count(_==k).toDouble
  val idf = Math.log((Ndocs.toDouble + 1.0) / (df.getOrElse(k,0).toDouble + 1.0))
  tf * idf
}
def topTerms(fn:String, k:Int=10): Seq[(String,Double)] =
  funcBags.getOrElse(fn, Seq.empty).distinct
    .map(t => (t, tfidf(fn,t))).sortBy(-_._2).take(k)

// Optional similarity to UCD terms (substring match over tokens)
def ucdMatchesFor(fn:String): Map[String, Seq[String]] = {
  if (ucdTerms.isEmpty) Map.empty
  else {
    val toks = funcBags.getOrElse(fn, Seq.empty).distinct
    val m = ucdTerms.map { u =>
      val uLow = u.toLowerCase
      val hits = toks.filter(t => t.contains(uLow) || uLow.contains(t))
      u -> hits
    }.filter(_._2.nonEmpty).toMap
    m
  }
}

val nameCueMap = cpg.method.name.l.flatMap { n =>
  val low = n.toLowerCase; nameClues.filter(low.contains).map(_ -> n)
}.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap
val stringCueMap = cpg.literal.code.l.flatMap { s =>
  val low = s.toLowerCase; nameClues.filter(low.contains).map(_ -> s)
}.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap

// ---------- Complexity & Workflow ----------
def cyclomaticApprox(m:Method): Int = {
  val cs = m.controlStructure.code.l
  1 + cs.count(isIfCode) + cs.count(isLoopCode) + cs.count(isSwitchCode)
}
val perFuncComplexity = definedMethods.map { m =>
  val rec = m.call.methodFullName.l.contains(m.fullName)
  (m.fullName, cyclomaticApprox(m), rec)
}

// ---------- Cross-Module / API ----------
def fileOfMethod(fn:String) = cpg.method.fullNameExact(fn).file.name.headOption
val crossModuleEdges = edges.filter { e =>
  (for { cf <- fileOfMethod(e.caller); tf <- fileOfMethod(e.callee) } yield baseName(cf) != baseName(tf)).getOrElse(false)
}
val publicApiList = publicMethods.map(_.fullName).distinct.sorted

// ---------- Security ----------
val unsafeCallsList = cpg.call.name.l.filter(knownUnsafe.contains).distinct.sorted
val degrees = functionFullNames.map(fn => fn -> (calleesByCaller.getOrElse(fn,Set()).size + callersByCallee.getOrElse(fn,Set()).size)).toMap
val highCouplingFunctions = degrees.filter(_._2 >= highCouplingThreshold).keys.toSeq.sorted

def unvalidatedInputHeuristic(m:Method): Int = {
  val ifs = m.controlStructure.isIf.code.l
  val hasElse = m.controlStructure.code.l.exists(_.trim.toLowerCase.startsWith("else"))
  val suspicious = ifs.count { s =>
    val low = s.toLowerCase
    inputNames.exists(low.contains) || low.contains("scanf") || low.contains("gets") || low.contains("cin")
  }
  if (!hasElse) suspicious else 0
}
val unvalidatedSum = definedMethods.map(unvalidatedInputHeuristic).sum

// ---------- Per-function enrichments ----------
case class FuncEnrich(
  fullName:String,
  file:String,
  line:Option[Int],
  branches:Int, prunedBranches:Int,
  cfg: CfgStats,
  topTerms: Seq[(String,Double)],
  ucdMatches: Map[String,Seq[String]]
)
val funcEnrich: Seq[FuncEnrich] = definedMethods.map { m =>
  val brs = branchInfos(m).filter(b => b.kind=="if" || b.kind=="switch")
  val pruned = brs.filter(b => meaningfulBranch(b, m))
  FuncEnrich(
    fullName = m.fullName,
    file = m.file.name.headOption.getOrElse(""),
    line = m.lineNumber.map(_.toInt),
    branches = brs.size,
    prunedBranches = pruned.size,
    cfg = cfgStats(m),
    topTerms = topTerms(m.fullName, 10),
    ucdMatches = ucdMatchesFor(m.fullName)
  )
}




// ---------- Per-function I/O hints & neighbors (Trace Seeds) ----------
def methodByFullName(fn:String): Option[Method] =
  cpg.method.fullNameExact(fn).headOption

def stringArgsOfIn(m:Method, names:Set[String]): Seq[String] =
  m.call.nameExact(names.toSeq:_*).argument.isLiteral.code.l
    .map(_.stripPrefix("\"").stripSuffix("\"")).distinct

def ioHintsFor(m:Method): Seq[String] = {
  val fLits = stringArgsOfIn(m, ioFile)
  val nLits = stringArgsOfIn(m, ioNet)
  val eLits = stringArgsOfIn(m, ioEnv)
  val cin   = m.call.nameExact("scanf","gets","getchar","cin").size
  val cout  = m.call.nameExact("printf","puts","putchar","cout","cerr").size
  val acc = scala.collection.mutable.ArrayBuffer[String]()
  if (fLits.nonEmpty) acc += ("file:" + fLits.take(3).mkString(","))
  if (nLits.nonEmpty) acc += ("net:"  + nLits.take(3).mkString(","))
  if (eLits.nonEmpty) acc += ("env:"  + eLits.take(3).mkString(","))
  if (cin  > 0) acc += s"console_in:$cin"
  if (cout > 0) acc += s"console_out:$cout"
  acc.toSeq
}

def neighborsOf(fn:String, k:Int=6): Seq[String] = {
  val in  = callersByCallee.getOrElse(fn, Set.empty)
  val out = calleesByCaller.getOrElse(fn, Set.empty)
  (in ++ out).toSeq.distinct.sorted.take(k)
}

val traceSeedsJson =
  "{"+
    "\"PerFunction\":"+
      funcEnrich.map { f =>
        val mopt = methodByFullName(f.fullName)
        val ioHs = mopt.map(ioHintsFor).getOrElse(Seq.empty)
        val neigh = neighborsOf(f.fullName)
        val lex   = f.topTerms.map(_._1).take(8)
        "{"+
          "\"fullName\":"+jstr(f.fullName)+","+
          "\"file\":"+jstr(f.file)+","+
          "\"line\":"+f.line.map(_.toString).getOrElse("null")+","+
          "\"lexical_terms\":"+jarr(lex)(jstr)+","+
          "\"io_hints\":"+jarr(ioHs)(jstr)+","+
          "\"cfg_hints\":"+
            jarr(Seq(
              s"branches:${f.branches}",
              s"pruned:${f.prunedBranches}",
              s"cfg_nodes:${f.cfg.nodes}",
              s"cfg_edges:${f.cfg.edges}",
              s"back_edges:${f.cfg.backEdges}",
              s"longest_acyclic:${f.cfg.longestAcyclic}",
              s"basic_segments:${f.cfg.basicSegments}"
            ))(jstr)+","+
          "\"neighbors\":"+jarr(neigh)(jstr)+
        "}"
      }.mkString("[",",","]")+
  "}"





// ---------- Not available from static CPG ----------
val notAvailable = Map(
  "RuntimeValuesOfVariables" -> "requires dynamic tracing / execution",
  "InteractionEventsCRUD"    -> "requires developer activity traces (VCS/IDE logs)",
  "ActorArtefactMapping"     -> "requires developer perspective logs",
  "CRIRelevanceCentrality"   -> "requires developer navigation traces"
)

// ---------- Assemble JSON (stack-safe: build by sections) ----------

// Section builders (kept small to avoid staging recursion)
val entryPointsJson =
  "{"+
    "\"Count\":"+jint(entryPointFullNames.size)+","+
    "\"List\":"+jarr(entryPointFullNames)(jstr)+
  "}"

val functionsJson =
  "{"+
    "\"Count\":"+jint(functionNames.size)+","+
    "\"List\":"+jarr(functionNames)(jstr)+","+
    "\"FullNames\":"+jarr(functionFullNames)(jstr)+","+
    "\"Signatures\":"+jarr(functionSigs)(jstr)+
  "}"

val callGraphJson =
  "{"+
    "\"Edges\":"+edges.map { e =>
      "{"+
        "\"caller\":"+jstr(e.caller)+","+
        "\"callee\":"+jstr(e.callee)+","+
        "\"file\":"+jstr(e.file)+","+
        "\"line\":"+(e.line.map(_.toString).getOrElse("null"))+
      "}"
    }.mkString("[",",","]")+
  "}"

val controlFlowJson =
  "{"+
    "\"Branch_Count\":"+jint(branchCount)+","+
    "\"Loop_Count\":"+jint(loopCount)+","+
    "\"Switch_Count\":"+jint(switchCount)+","+
    "\"ErrorExit.Paths\":"+jint(errorExitTotal)+
  "}"

val ioJson =
  "{"+
    "\"Console_Input_Uses\":"+jint(consoleInCount)+","+
    "\"Console_Output_Uses\":"+jint(consoleOutCount)+","+
    "\"File_IO_Uses\":"+jint(fileIOCount)+","+
    "\"Network_IO_Uses\":"+jint(netIOCount)+","+
    "\"EnvVar_Uses\":"+jint(envIOCount)+
  "}"

val dataArtifactsJson =
  "{"+
    "\"CLI.Args.Usage\":"+jint(cliArgsUse)+","+
    "\"SharedArtifacts\":"+jarr(sharedArtifacts)(jstr)+","+
    "\"Files.Touched\":"+jarr(filesTouched)(jstr)+","+
    "\"Network.Endpoints\":"+jarr(networkEndpoints)(jstr)+
  "}"

val nameTextCuesJson = {
  val nameCuesLocal = cpg.method.name.l.flatMap { n =>
    val low = n.toLowerCase; nameClues.filter(low.contains).map(_ -> n)
  }.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap

  val stringCuesLocal = cpg.literal.code.l.flatMap { s =>
    val low = s.toLowerCase; nameClues.filter(low.contains).map(_ -> s)
  }.groupBy(_._1).view.mapValues(_.map(_._2).distinct.sorted).toMap

  "{"+
    "\"NameCues.Match\":"+jmap(nameCuesLocal)(vs => jarr(vs)(jstr))+","+
    "\"StringCues.Match\":"+jmap(stringCuesLocal)(vs => jarr(vs)(jstr))+
  "}"
}

val complexityWorkflowJson = {
  val perFunc =
    perFuncComplexity.map { case (fn,cpx,rec) =>
      "{"+
        "\"function\":"+jstr(fn)+","+
        "\"ComplexityApprox\":"+jint(cpx)+","+
        "\"Recursion.Flag\":"+jbool(rec)+
      "}"
    }.mkString("[",",","]")

  // longest path from entry (recomputed section-locally)
  val adj = calleesByCaller.view.mapValues(_.toSet).toMap
  val maxFromEntry = entryPointFullNames.map { n =>
    val memo = scala.collection.mutable.Map.empty[String, Int]
    def lf(x:String, seen:Set[String]): Int =
      memo.getOrElseUpdate(x, adj.getOrElse(x, Set.empty).filterNot(seen)
        .map(y => 1 + lf(y, seen + x)).foldLeft(0)(Math.max))
    n -> lf(n, Set.empty)
  }.toMap

  "{"+
    "\"ComplexityApprox.PerFunction\":"+perFunc+","+
    "\"Max_Path_Length_From_Entry\":"+jmap(maxFromEntry)(v => jint(v))+
  "}"
}

val crossModuleApiJson =
  "{"+
    "\"CrossModule.Calls\":"+jint(crossModuleEdges.size)+","+
    "\"PublicAPI.Count\":"+jint(publicApiList.size)+","+
    "\"PublicAPI.List\":"+jarr(publicApiList)(jstr)+
  "}"

val securityJson =
  "{"+
    "\"UnsafeCalls.List\":"+jarr(unsafeCallsList)(jstr)+","+
    "\"UnvalidatedInput.Heuristics\":"+jint(unvalidatedSum)+","+
    "\"HighCoupling.Functions\":"+jarr(highCouplingFunctions)(jstr)+
  "}"

val researchBrcgJson =
  "{"+
    "\"BranchStatements.PerFunction\":"+funcEnrich.map { f =>
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"branches\":"+jint(f.branches)+","+
        "\"prunedBranches\":"+jint(f.prunedBranches)+
      "}"
    }.mkString("[",",","]")+","+
    "\"ProcedureCalls.Edges\":"+edges.map { e =>
      "{"+
        "\"caller\":"+jstr(e.caller)+","+
        "\"callee\":"+jstr(e.callee)+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchCfgJson =
  "{"+
    "\"PerFunction\":"+funcEnrich.map { f =>
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"nodes\":"+jint(f.cfg.nodes)+","+
        "\"edges\":"+jint(f.cfg.edges)+","+
        "\"backEdgesHeuristic\":"+jint(f.cfg.backEdges)+","+
        "\"longestAcyclicPathApprox\":"+jint(f.cfg.longestAcyclic)+","+
        "\"basicSegments\":"+jint(f.cfg.basicSegments)+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchDomainTermsJson =
  "{"+
    "\"TopTFIDF.PerFunction\":"+funcEnrich.map { f =>
      val tops = f.topTerms.map { case (t,score) =>
        "{"+
          "\"term\":"+jstr(t)+","+
          "\"score\":"+jstr("%.5f".format(score))+
        "}"
      }.mkString("[",",","]")
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"TopTerms\":"+tops+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchCommentsJson =
  "{"+
    "\"HasComments.PerFunction\":"+definedMethods.map { m =>
      "{"+
        "\"function\":"+jstr(m.fullName)+","+
        "\"count\":"+jint(functionComments(m).size)+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchRelationsJson = {
  val inh = cpg.typeDecl.l.map { td =>
    "{"+
      "\"type\":"+jstr(Option(td.fullName).getOrElse(""))+","+
      "\"inherits\":"+jarr(td.inheritsFromTypeFullName.l.distinct.sorted)(jstr)+
    "}"
  }.mkString("[",",","]")

  val varToFns = jmap(sharedVarToFuncs.map { case (k,v) => k -> v.toSeq.sorted })(vs => jarr(vs)(jstr))

  "{"+
    "\"Inheritance\":"+inh+","+
    "\"SharedGlobals\":{"+
      "\"Variables\":"+jarr(sharedVarToFuncs.keys.toSeq.sorted)(jstr)+","+
      "\"VarToFunctions\":"+varToFns+
    "}"+
  "}"
}

val researchSimilarityJson =
  "{"+
    "\"Enabled\":"+jbool(ucdTerms.nonEmpty)+","+
    "\"UCDElements\":"+jarr(ucdTerms)(jstr)+","+
    "\"Matches.PerFunction\":"+funcEnrich.map { f =>
      "{"+
        "\"function\":"+jstr(f.fullName)+","+
        "\"matches\":"+jmap(f.ucdMatches)(vs => jarr(vs)(jstr))+
      "}"
    }.mkString("[",",","]")+
  "}"

val researchNotAvailJson = jmap(notAvailable)(jstr)

val researchJson =
  "{"+
    "\"BRCG\":"+researchBrcgJson+","+
    "\"CFGTraversal\":"+researchCfgJson+","+
    "\"DomainTerms\":"+researchDomainTermsJson+","+
    "\"Comments\":"+researchCommentsJson+","+
    "\"Relations\":"+researchRelationsJson+","+
    "\"SimilarityToUCD\":"+researchSimilarityJson+","+
    "\"NotAvailableFromStaticCPG\":"+researchNotAvailJson+
  "}"

val top =
  "{"+
    "\"EntryPoints\":"+entryPointsJson+","+
    "\"Functions\":"+functionsJson+","+
    "\"CallGraph\":"+callGraphJson+","+
    "\"ControlFlow\":"+controlFlowJson+","+
    "\"IO\":"+ioJson+","+
    "\"DataArtifacts\":"+dataArtifactsJson+","+
    "\"NameTextCues\":"+nameTextCuesJson+","+
    "\"ComplexityWorkflow\":"+complexityWorkflowJson+","+
    "\"CrossModuleAPI\":"+crossModuleApiJson+","+
    "\"Security\":"+securityJson+","+
    "\"TraceSeeds\":"+traceSeedsJson+","+ 
    "\"ResearchMetrics\":"+researchJson+
  "}"


Files.write(Paths.get(outPath), top.getBytes(StandardCharsets.UTF_8))
println("Wrote JSON to: "+outPath)